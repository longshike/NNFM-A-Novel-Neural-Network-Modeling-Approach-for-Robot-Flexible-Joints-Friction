#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>
#include <map>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include "examples_common.h"

namespace {
template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}

// 平滑启动函数
double smoothStart(double t, double start_time, double duration) {
    if (t < start_time) return 0.0;
    if (t > start_time + duration) return 1.0;
    double x = (t - start_time) / duration;
    // 使用平滑的缓动函数
    return x * x * x * (x * (x * 6 - 15) + 10);
}

// 平滑停止函数
double smoothStop(double t, double stop_start_time, double stop_duration) {
    if (t < stop_start_time) return 1.0;
    if (t > stop_start_time + stop_duration) return 0.0;
    double x = (t - stop_start_time) / stop_duration;
    // 平滑的缓动函数
    return 1.0 - (x * x * x * (x * (x * 6 - 15) + 10));
}

// 摩擦模型参数结构体
struct FrictionParams {
  // CV模型参数
  struct CVParams {
    std::array<double, 7> Fc = {{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
    std::array<double, 7> fv = {{0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01}};
  };
  
  // Dahl模型参数
  struct DahlParams {
    std::array<double, 7> sigma0 = {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}};
    std::array<double, 7> Fc = {{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
    std::array<double, 7> beta = {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
    double dt = 0.001;
  };
  
  // Stribeck模型参数
  struct StribeckParams {
    std::array<double, 7> Fc = {{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
    std::array<double, 7> Fs = {{0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15}};
    std::array<double, 7> fv = {{0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01}};
    std::array<double, 7> vs = {{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
    std::array<double, 7> alpha = {{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0}};
  };
  
  // LuGre模型参数
  struct LuGreParams {
    std::array<double, 7> sigma0 = {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}};
    std::array<double, 7> sigma1 = {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}};
    std::array<double, 7> sigma2 = {{0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01}};
    std::array<double, 7> Fc = {{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
    std::array<double, 7> Fs = {{0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15}};
    std::array<double, 7> vs = {{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
  };
  
  // NNFM模型参数
  struct NNFMParams {
    std::array<double, 7> f_s = {{0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15}};
    std::array<double, 7> w_f = {{0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6}};
    std::array<std::array<double, 8>, 7> w_odd;
    std::array<std::array<double, 8>, 7> c_odd;
    
    NNFMParams() {
      // 初始化NNFM参数
      std::array<double, 8> base_w = {{0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05}};
      std::array<double, 8> base_c = {{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}};
      
      for (int i = 0; i < 7; ++i) {
        w_odd[i] = base_w;
        c_odd[i] = base_c;
      }
    }
  };
  
  CVParams cv_params;
  DahlParams dahl_params;
  StribeckParams stribeck_params;
  LuGreParams lugre_params;
  NNFMParams nnfm_params;
};

// 摩擦模型状态
struct FrictionStates {
  // Dahl模型状态
  std::array<double, 7> dahl_z = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  
  // LuGre模型状态
  std::array<double, 7> lugre_z = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  
  // NNFM模型状态
  std::array<double, 7> nnfm_HH_prev = {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
};

// 摩擦模型计算类
class FrictionModels {
private:
  FrictionParams params_;
  FrictionStates states_;

public:
  FrictionModels(const FrictionParams& params) : params_(params) {}
  
  // CV摩擦模型
  double cv_friction(int joint_id, double dq) {
    return params_.cv_params.Fc[joint_id] * sign(dq) + params_.cv_params.fv[joint_id] * dq;
  }
  
  // Dahl摩擦模型
  double dahl_friction(int joint_id, double dq, double dt) {
    double sigma0 = params_.dahl_params.sigma0[joint_id];
    double Fc = params_.dahl_params.Fc[joint_id];
    double beta = params_.dahl_params.beta[joint_id];
    double& z = states_.dahl_z[joint_id];
    
    // 计算中间变量γ
    double gamma;
    if (std::abs(dq) > 1e-6) {
      gamma = 1.0 - sign(dq) * sigma0 * z / Fc;
    } else {
      gamma = 1.0;
    }
    
    // 计算刷毛挠度变化率?
    double dz_dt;
    if (std::abs(gamma) < 1e-6) {
      dz_dt = 0.0;
    } else {
      dz_dt = dq * sign(gamma) * std::pow(std::abs(gamma), beta);
    }
    
    // 更新刷毛挠度z（欧拉积分）
    z = z + dz_dt * dt;
    
    // 物理约束：挠度z的绝对值不超过Fc/sigma0
    double z_max = Fc / sigma0;
    z = std::max(std::min(z, z_max), -z_max);
    
    // 计算摩擦扭矩
    return sigma0 * z;
  }
  
  // Stribeck摩擦模型
  double stribeck_friction(int joint_id, double dq) {
    double Fc = params_.stribeck_params.Fc[joint_id];
    double Fs = params_.stribeck_params.Fs[joint_id];
    double fv = params_.stribeck_params.fv[joint_id];
    double vs = params_.stribeck_params.vs[joint_id];
    double alpha = params_.stribeck_params.alpha[joint_id];
    
    // 计算Stribeck摩擦
    double stribeck_term = Fc + (Fs - Fc) * std::exp(-std::pow(std::abs(dq) / vs, alpha));
    return fv * dq + sign(dq) * stribeck_term;
  }
  
  // LuGre摩擦模型（稳态近似）
  double lugre_friction_steady(int joint_id, double dq) {
    double sigma0 = params_.lugre_params.sigma0[joint_id];
    double sigma1 = params_.lugre_params.sigma1[joint_id];
    double sigma2 = params_.lugre_params.sigma2[joint_id];
    double Fc = params_.lugre_params.Fc[joint_id];
    double Fs = params_.lugre_params.Fs[joint_id];
    double vs = params_.lugre_params.vs[joint_id];
    
    // 计算LuGre摩擦（稳态近似）
    double g_dq = (Fc + (Fs - Fc) * std::exp(-std::pow(std::abs(dq) / vs, 2.0))) / sigma0;
    double z_steady = g_dq * sign(dq);
    double dz_dt_steady = 0.0;  // 稳态下dz/dt=0
    
    return sigma0 * z_steady + sigma1 * dz_dt_steady + sigma2 * dq;
  }
  
  // LuGre摩擦模型（动态版本）
  double lugre_friction_dynamic(int joint_id, double dq, double dt) {
    double sigma0 = params_.lugre_params.sigma0[joint_id];
    double sigma1 = params_.lugre_params.sigma1[joint_id];
    double sigma2 = params_.lugre_params.sigma2[joint_id];
    double Fc = params_.lugre_params.Fc[joint_id];
    double Fs = params_.lugre_params.Fs[joint_id];
    double vs = params_.lugre_params.vs[joint_id];
    double& z = states_.lugre_z[joint_id];
    
    // 计算g(dq)
    double g_dq = (Fc + (Fs - Fc) * std::exp(-std::pow(std::abs(dq) / vs, 2.0))) / sigma0;
    
    // 更新内部状态z
    double dz_dt;
    if (std::abs(dq) > 1e-6) {
      dz_dt = dq - (std::abs(dq) / g_dq) * z;
    } else {
      dz_dt = - (1.0 / g_dq) * z;
    }
    
    z = z + dz_dt * dt;
    
    // 计算摩擦扭矩
    return sigma0 * z + sigma1 * dz_dt + sigma2 * dq;
  }
  
  // NNFM摩擦模型
  double nnfm_friction(int joint_id, double dq) {
    double f_s = params_.nnfm_params.f_s[joint_id];
    double w_f = params_.nnfm_params.w_f[joint_id];
    auto& w_odd = params_.nnfm_params.w_odd[joint_id];
    auto& c_odd = params_.nnfm_params.c_odd[joint_id];
    double& HH_prev = states_.nnfm_HH_prev[joint_id];
    
    int n = 8;  // 奇数阶项数量
    
    // 非光滑部分
    double nonsmooth_part = f_s * sign(dq);
    
    // 光滑部分计算
    double smooth_part = 0.0;
    for (int j = 0; j < n; ++j) {
      int order = 2 * j + 1;  // 奇数阶: 1, 3, 5, ...
      
      // 改进的激活函数组合
      double linear_component = c_odd[j] * std::pow(dq, order);
      double tanh_component = std::tanh(linear_component);
      
      // 添加sigmoid变换增加非线性
      double sigmoid_component = 1.0 / (1.0 + std::exp(-0.5 * linear_component));
      
      // 组合多个激活函数
      double combined_activation = 0.7 * tanh_component + 0.3 * sigmoid_component;
      
      double term = w_odd[j] * combined_activation;
      smooth_part += term;
    }
    
    // 添加动态反馈
    double HH_current = w_f * HH_prev + smooth_part;
    states_.nnfm_HH_prev[joint_id] = HH_current;
    
    // 总摩擦力矩
    return HH_current + nonsmooth_part;
  }
  
  // 获取所有摩擦模型的预测结果
  std::array<double, 5> getAllFrictionPredictions(int joint_id, double dq, double dt) {
    std::array<double, 5> predictions;
    
    predictions[0] = cv_friction(joint_id, dq);
    predictions[1] = dahl_friction(joint_id, dq, dt);
    predictions[2] = stribeck_friction(joint_id, dq);
    predictions[3] = lugre_friction_dynamic(joint_id, dq, dt);
    predictions[4] = nnfm_friction(joint_id, dq);
    
    return predictions;
  }
  
private:
  double sign(double x) {
    return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
  }
};

}  // anonymous namespace

int main(int argc, char** argv) {
  // Check whether the required arguments were passed.
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }
  // 设置斜八字形轨迹参数 - 修改为3分钟
  const double amplitude = 0.03;       // 基础幅度
  //const double frequency = 0.02;       // 慢速、运动频率
  //const double frequency = 0.04;       // 中速、运动频率
  const double frequency = 0.08;       // 快速、运动频率
  const double run_time = 120.0;       // 修改为180秒（3分钟）
  const double start_up_time = 4.0;    // 启动时间
  const double stop_time = 4.0;        // 停止时间
  
  // 斜八字轨迹参数 - XY平面八字 + Z方向倾斜
  const double xy_amplitude = 0.1;    // XY平面八字幅度
  const double z_amplitude = 0.03;     // Z方向幅度
  //const double z_frequency = 0.04;     // 慢速、Z方向频率 (XY频率的2倍)
  //const double z_frequency = 0.08;     // 中速、Z方向频率 (XY频率的2倍)
  const double z_frequency = 0.16;     // 快速、Z方向频率 (XY频率的2倍)
  const double z_phase_offset = M_PI/4; // Z方向相位偏移
  
  // 设置打印速度
  const double print_rate = 500.0;

  double time = 0.0;  //设置时间为零

  // 初始化摩擦模型参数
  FrictionParams friction_params;
  // 这里可以设置具体的摩擦参数，使用从MATLAB辨识得到的参数
  
  FrictionModels friction_models(friction_params);

  // 初始化打印进程中的初始化数据
  struct {
    std::mutex mutex;  //进程
    bool has_data;
    double time_save;  //定义时间
    franka::RobotState robot_state;
    std::array<double, 7> tau_J_compensate;  //补偿重力和零偏后的测量力矩
    std::array<std::array<double, 5>, 7> friction_predictions;  // 7个关节，每个关节5个模型的预测
  } print_data{};
  std::atomic_bool running{true};
  std::atomic_bool init_csv{false};

  // Start print thread.开始启动打印进程
  std::thread print_thread([print_rate, &print_data, &running, &init_csv]() {
    if (!init_csv) {
      //初始化文件csv文件
      std::ofstream outFile;                                          // 创建流对象
      outFile.open("test_data_xy_figure8_with_z_friction_models.csv", std::ios::out);  //修改文件名
      outFile << "time,q1,q2,q3,q4,q5,q6,q7,dq1,dq2,dq3,dq4,dq5,dq6,dq7" << ','
              << "ddq1,ddq2,ddq3,ddq4,ddq5,ddq6,ddq7" << ','
              << "tau_measured_compensate1,tau_measured_compensate2,tau_measured_compensate3,tau_measured_compensate4,"
                 "tau_measured_compensate5,tau_measured_compensate6,tau_measured_compensate7" << ','
              << "CV_friction1,CV_friction2,CV_friction3,CV_friction4,CV_friction5,CV_friction6,CV_friction7" << ','
              << "Dahl_friction1,Dahl_friction2,Dahl_friction3,Dahl_friction4,Dahl_friction5,Dahl_friction6,Dahl_friction7" << ','
              << "Stribeck_friction1,Stribeck_friction2,Stribeck_friction3,Stribeck_friction4,Stribeck_friction5,Stribeck_friction6,Stribeck_friction7" << ','
              << "LuGre_friction1,LuGre_friction2,LuGre_friction3,LuGre_friction4,LuGre_friction5,LuGre_friction6,LuGre_friction7" << ','
              << "NNFM_friction1,NNFM_friction2,NNFM_friction3,NNFM_friction4,NNFM_friction5,NNFM_friction6,NNFM_friction7"
              << std::endl;  // 初始化第一行
      outFile.close();       //关闭文件
      init_csv = true;
    }

    auto last_print_time = std::chrono::steady_clock::now();
    const auto print_interval =
        std::chrono::milliseconds(static_cast<int>((1.0 / print_rate * 1000.0)));

    while (running) {
      auto current_time = std::chrono::steady_clock::now();
      if (current_time - last_print_time < print_interval) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
      last_print_time = current_time;

      // 试图锁住数据，防止读写发生堵塞
      if (print_data.mutex.try_lock()) {
        if (print_data.has_data) {
          // 将结果打印到屏幕上
          std::cout << "Time: " << print_data.time_save
                    << " tau_measured_compensate [Nm]: " << print_data.tau_J_compensate
                    << std::endl;
          // 将结果保存到csv中
          if (init_csv) {
            std::ofstream outFile;  // 创建流对象
            outFile.open("test_data_xy_figure8_with_z_friction_models.csv", std::ios::app);
            outFile << print_data.time_save << ',' 
                    << print_data.robot_state.q[0] << ',' << print_data.robot_state.q[1] << ',' 
                    << print_data.robot_state.q[2] << ',' << print_data.robot_state.q[3] << ',' 
                    << print_data.robot_state.q[4] << ',' << print_data.robot_state.q[5] << ',' 
                    << print_data.robot_state.q[6] << ','
                    << print_data.robot_state.dq[0] << ',' << print_data.robot_state.dq[1] << ',' 
                    << print_data.robot_state.dq[2] << ',' << print_data.robot_state.dq[3] << ',' 
                    << print_data.robot_state.dq[4] << ',' << print_data.robot_state.dq[5] << ',' 
                    << print_data.robot_state.dq[6] << ','
                    << print_data.robot_state.ddq_d[0] << ',' << print_data.robot_state.ddq_d[1] << ',' 
                    << print_data.robot_state.ddq_d[2] << ',' << print_data.robot_state.ddq_d[3] << ',' 
                    << print_data.robot_state.ddq_d[4] << ',' << print_data.robot_state.ddq_d[5] << ',' 
                    << print_data.robot_state.ddq_d[6] << ','
                    << print_data.tau_J_compensate[0] << ',' << print_data.tau_J_compensate[1] << ',' 
                    << print_data.tau_J_compensate[2] << ',' << print_data.tau_J_compensate[3] << ',' 
                    << print_data.tau_J_compensate[4] << ',' << print_data.tau_J_compensate[5] << ',' 
                    << print_data.tau_J_compensate[6] << ',';
            
            // 写入CV模型预测
            for (int i = 0; i < 7; ++i) {
              outFile << print_data.friction_predictions[i][0];
              if (i < 6) outFile << ',';
            }
            outFile << ',';
            
            // 写入Dahl模型预测
            for (int i = 0; i < 7; ++i) {
              outFile << print_data.friction_predictions[i][1];
              if (i < 6) outFile << ',';
            }
            outFile << ',';
            
            // 写入Stribeck模型预测
            for (int i = 0; i < 7; ++i) {
              outFile << print_data.friction_predictions[i][2];
              if (i < 6) outFile << ',';
            }
            outFile << ',';
            
            // 写入LuGre模型预测
            for (int i = 0; i < 7; ++i) {
              outFile << print_data.friction_predictions[i][3];
              if (i < 6) outFile << ',';
            }
            outFile << ',';
            
            // 写入NNFM模型预测
            for (int i = 0; i < 7; ++i) {
              outFile << print_data.friction_predictions[i][4];
              if (i < 6) outFile << ',';
            }
            outFile << std::endl;
            
            outFile.close();  //关闭文件
          }
          print_data.has_data = false;
        }
        print_data.mutex.unlock();
      }
    }
  });

  try {
    // Connect to robot.连接上机械臂
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);  //设置机械臂的默认参数，包括碰撞扭矩的上下限，碰撞力的上下限

    //首先将机械臂移至一个合适关节位置
    std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    MotionGenerator motion_generator(0.3, q_goal);  //降低运动速度
    std::cout << "WARNING: This example will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    robot.control(motion_generator);  //调用运动生成器
    std::cout << "Finished moving to initial joint configuration."
              << std::endl;  //完成初始关节位置配置

    // 等待稳定
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 设置碰撞检测
    robot.setCollisionBehavior(
        {{60.0, 60.0, 60.0, 60.0, 40.0, 40.0, 40.0}}, {{60.0, 60.0, 60.0, 60.0, 40.0, 40.0, 40.0}},
        {{60.0, 60.0, 60.0, 60.0, 40.0, 40.0}}, {{60.0, 60.0, 60.0, 60.0, 40.0, 40.0}});

    // 下载运动学和动力学模型
    franka::Model model = robot.loadModel();

    //进行传感器初始零偏值计算
    franka::RobotState initial_state = robot.readOnce();  //获取初始位置
    Eigen::VectorXd initial_tau_ext(7);
    // 获取初始化重力补偿项
    std::array<double, 7> gravity_array = model.gravity(initial_state);
    Eigen::Map<Eigen::Matrix<double, 7, 1>> initial_tau_measured(initial_state.tau_J.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1>> initial_gravity(gravity_array.data());
    // 传感器测量数据扣除重力补偿，获取传感器零偏
    initial_tau_ext = initial_tau_measured - initial_gravity;

    /**斜八字形轨迹部分 - 使用笛卡尔速度控制**/
    auto cartesian_pose_callback = [xy_amplitude, z_amplitude, frequency, z_frequency, 
                                   run_time, start_up_time, stop_time, z_phase_offset,
                                   &time, &running](
                                       const franka::RobotState&,
                                       franka::Duration period) -> franka::CartesianVelocities {
      time += period.toSec();  //更新时间

      // 计算实际运动结束时间（考虑停止时间）
      const double motion_end_time = run_time - stop_time;

      // 如果超过运行时间，发送停止命令
      if (time >= run_time) {
        running.store(false);
        std::cout << std::endl << "Finished 3-minute XY figure-8 with Z motion." << std::endl;
        // 明确创建 CartesianVelocities 对象
        franka::CartesianVelocities zero_velocities = {{0, 0, 0, 0, 0, 0}};
        return franka::MotionFinished(zero_velocities);
      }

      double angle = 2.0 * M_PI * frequency * time;
      
      // XY平面八字形轨迹速度计算
      // 标准八字形参数方程：x = A*sin(ωt), y = A*sin(2ωt)
      double base_v_x = 2.0 * M_PI * frequency * xy_amplitude * std::cos(angle);           // dx/dt = Aω*cos(ωt)
      double base_v_y = 4.0 * M_PI * frequency * xy_amplitude * std::cos(2.0 * angle);     // dy/dt = 2Aω*cos(2ωt)
      
      // Z方向运动 - 与XY平面运动协调
      double base_v_z = 2.0 * M_PI * z_frequency * z_amplitude * std::cos(2.0 * M_PI * z_frequency * time + z_phase_offset);

      // 应用平滑启动和平滑停止
      double ease_factor = smoothStart(time, 0.0, start_up_time);
      
      if (time > motion_end_time) {
        double stop_factor = smoothStop(time, motion_end_time, stop_time);
        ease_factor *= stop_factor;
      }
      
      double v_x = base_v_x * ease_factor;
      double v_y = base_v_y * ease_factor;
      double v_z = base_v_z * ease_factor;

      franka::CartesianVelocities output = {{v_x, v_y, v_z, 0.0, 0.0, 0.0}};

      // 定期输出进度信息和轨迹信息
      if (static_cast<int>(time) % 30 == 0 && static_cast<int>(time) > 0) {
        int minutes = static_cast<int>(time) / 60;
        int seconds = static_cast<int>(time) % 60;
        
        // 计算当前位置（积分速度得到位置）
        static double x_pos = 0, y_pos = 0, z_pos = 0;
        x_pos += v_x * period.toSec();
        y_pos += v_y * period.toSec();
        z_pos += v_z * period.toSec();
        
        std::cout << "Progress: " << minutes << "m " << seconds << "s / 3m 0s" 
                  << " Pos(X,Y,Z): (" << x_pos << ", " << y_pos << ", " << z_pos << ")"
                  << " Vel(X,Y,Z): (" << v_x << ", " << v_y << ", " << v_z << ")" << std::endl;
      }

      return output;
    };

    // 设置关节阻抗控制增益
    const std::array<double, 7> k_gains = {{300.0, 300.0, 300.0, 300.0, 150.0, 80.0, 20.0}};
    const std::array<double, 7> d_gains = {{20.0, 20.0, 20.0, 20.0, 10.0, 8.0, 3.0}};

    // 定义关节力矩控制回路的回调函数
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback =
            [&time, &print_data, &model, &friction_models, k_gains, d_gains, initial_tau_ext](
                const franka::RobotState& state, franka::Duration period) -> franka::Torques {
      time += period.toSec();  // 更新时间
      
      // 读取当前模型中的哥氏矩阵\重力矩阵\实时扭矩测量值
      std::array<double, 7> coriolis = model.coriolis(state);
      std::array<double, 7> gravity_array1 = model.gravity(state);

      // 将上述获得的变量转化为矩阵形式
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis_calculate(coriolis.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> gravity_calculate(gravity_array1.data());
      Eigen::Map<const Eigen::Matrix<double, 7, 1>> tau_measured(state.tau_J.data());

      //计算实时的测量力矩
      Eigen::VectorXd tau_ext(7);
      tau_ext = tau_measured - gravity_calculate - initial_tau_ext;

      std::array<double, 7> tau_d_calculated;  //定义计算的控制力矩输出
      for (size_t i = 0; i < 7; i++) {
        tau_d_calculated[i] =
            k_gains[i] * (state.q_d[i] - state.q[i]) - d_gains[i] * state.dq[i] + coriolis[i];
      }

      std::array<double, 7> tau_d_rate_limited =
          franka::limitRate(franka::kMaxTorqueRate, tau_d_calculated, state.tau_J_d);

      //将计算的结果由矩阵形式转化为array
      std::array<double, 7> tau_ext_array{};
      //转回列表形式
      Eigen::VectorXd::Map(&tau_ext_array[0], 7) = tau_ext;
      
      // 计算所有摩擦模型的预测
      std::array<std::array<double, 5>, 7> friction_predictions;
      double dt = period.toSec();
      
      for (int i = 0; i < 7; ++i) {
        friction_predictions[i] = friction_models.getAllFrictionPredictions(i, state.dq[i], dt);
      }

      // Update data to print.更新数据到print结构
      if (print_data.mutex.try_lock()) {
        print_data.has_data = true;
        print_data.time_save = time;
        print_data.robot_state = state;
        print_data.tau_J_compensate = tau_ext_array;
        print_data.friction_predictions = friction_predictions;
        print_data.mutex.unlock();
      }

      // 发送控制力矩指令
      return tau_d_rate_limited;
    };

    std::cout << "Starting 3-MINUTE XY FIGURE-8 with Z MOTION trajectory..." << std::endl;
    std::cout << "Total duration: 180 seconds (3 minutes)" << std::endl;
    std::cout << "XY Figure-8 amplitude: " << xy_amplitude << "m" << std::endl;
    std::cout << "Z motion amplitude: " << z_amplitude << "m" << std::endl;
    std::cout << "Base frequency: " << frequency << "Hz" << std::endl;
    std::cout << "Z frequency: " << z_frequency << "Hz" << std::endl;
    std::cout << "The robot will trace a figure-8 in XY plane with vertical Z motion." << std::endl;
    std::cout << "This creates a 3D slanted/spiral figure-8 pattern." << std::endl;
    std::cout << "Friction model comparison data will be saved to: test_data_xy_figure8_with_z_friction_models.csv" << std::endl;

    // 开启实时控制回路
    robot.control(impedance_control_callback, cartesian_pose_callback);

  } catch (const franka::Exception& ex) {
    running.store(false);
    std::cerr << "Franka exception: " << ex.what() << std::endl;
  } catch (const std::exception& ex) {
    running.store(false);
    std::cerr << "Exception: " << ex.what() << std::endl;
  }

  if (print_thread.joinable()) {
    print_thread.join();
  }

  std::cout << "3-minute XY figure-8 with Z motion trajectory program finished." << std::endl;
  std::cout << "Friction model comparison data saved to: test_data_xy_figure8_with_z_friction_models.csv" << std::endl;
  return 0;
}