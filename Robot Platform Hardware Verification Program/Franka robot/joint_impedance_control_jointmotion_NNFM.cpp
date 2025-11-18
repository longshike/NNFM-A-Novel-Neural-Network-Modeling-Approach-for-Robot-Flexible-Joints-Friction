//该程序在圆形阻抗控制的基础上,加入控制关节运动的轨迹，用于进行关节运动的摩擦对比
// 添加NNFM摩擦补偿
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
#include <memory>
#include <algorithm>

#include <Eigen/Core>

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
}  // anonymous namespace

// NNFM 类定义
class NNFM {
public:
    struct Config {
        int n;  // 网络宽度（奇数阶项数量）
        std::array<double, 4> learning_rates; // [η1, η2, η3, η4]
        
        // 默认构造函数
        Config() : n(5), learning_rates({0.01, 0.01, 0.001, 0.001}) {}
    };

    struct Model {
        int joint_index;
        Config config;
        double f_s;  // 符号函数幅值
        double w_f;  // 动态反馈权重
        std::vector<double> w_odd;  // 奇数阶项权重
        std::vector<double> c_odd;  // 奇数阶项系数
        double HH_prev;  // 历史状态
        bool is_initialized;
        int optimization_step;
        
        // 默认构造函数
        Model() : joint_index(0), f_s(0.1), w_f(0.5), HH_prev(0.0), is_initialized(false), optimization_step(0) {}
    };

    // 构造函数
    NNFM(int joint_id, const Config& config = Config()) {
        model_.joint_index = joint_id;
        model_.config = config;
        model_.f_s = 0.1;
        model_.w_f = 0.5;
        model_.HH_prev = 0.0;
        model_.is_initialized = false;
        model_.optimization_step = 0;
        
        // 初始化奇数阶项参数
        model_.w_odd.resize(config.n, 0.1);
        model_.c_odd.resize(config.n, 0.1);
        model_.is_initialized = true;
    }
    
    // 前向传播预测摩擦力矩
    double forward(double dq) {
        int n = model_.config.n;
        
        // 非光滑部分: f_s * sgn(dq)
        double nonsmooth_part = model_.f_s * sign_custom(dq);
        
        // 光滑部分计算
        double smooth_part = 0.0;
        for (int j = 0; j < n; j++) {
            int order = 2 * j + 1;  // 奇数阶: 1, 3, 5, ...
            double term = model_.w_odd[j] * tanh_custom(model_.c_odd[j] * std::pow(dq, order));
            smooth_part += term;
        }
        
        // 添加动态反馈
        double HH_current = model_.w_f * model_.HH_prev + smooth_part;
        model_.HH_prev = HH_current;  // 更新历史状态
        
        // 总摩擦力矩
        return HH_current + nonsmooth_part;
    }
    
    // 在线参数更新
    void update(double dq, double tau_measured, double tau_pred, double error, int step) {
        const auto& learning_rates = model_.config.learning_rates;
        int n = model_.config.n;
        
        // 更新 f_s (η1)
        model_.f_s += learning_rates[0] * error * sign_custom(dq);
        
        // 更新 w_odd (η2)
        for (int j = 0; j < n; j++) {
            int order = 2 * j + 1;
            double tanh_term = tanh_custom(model_.c_odd[j] * std::pow(dq, order));
            model_.w_odd[j] += learning_rates[1] * error * tanh_term;
        }
        
        // 更新 c_odd (η3)
        for (int j = 0; j < n; j++) {
            int order = 2 * j + 1;
            double tanh_input = model_.c_odd[j] * std::pow(dq, order);
            double tanh_derivative = 1.0 - std::pow(tanh_custom(tanh_input), 2);
            model_.c_odd[j] += learning_rates[2] * error * model_.w_odd[j] * 
                              tanh_derivative * std::pow(dq, order);
        }
        
        // 更新 w_f (η4)
        model_.w_f += learning_rates[3] * error * model_.HH_prev;
        
        // 确保参数在合理范围内
        model_.f_s = std::max(0.001, std::min(10.0, model_.f_s));
        model_.w_f = std::max(0.0, std::min(1.0, model_.w_f));
        
        model_.optimization_step = step;
    }
    
    // 获取当前模型状态
    Model getModel() const { return model_; }
    
private:
    Model model_;
    
    // 计算双曲正切函数
    double tanh_custom(double x) {
        return std::tanh(x);
    }
    
    // 计算符号函数
    double sign_custom(double x) {
        return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0);
    }
};

int main(int argc, char** argv) {
  // Check whether the required arguments were passed.
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }
  
  // 控制模式选择
  bool use_nnfm_compensation = true;  // 设置为true使用NNFM补偿，false不使用
  
  // 设置初始化轨迹参数
  const double time_max = 1.0;
  const double omega_max = 0.4;
  const double run_time = 20.0;  //运行时间
  // 设置打印速度
  const double print_rate = 1000.0;

  double time = 0.0;  //设置时间为零

  // 初始化NNFM模型
  std::vector<std::unique_ptr<NNFM>> nnfm_models;
  NNFM::Config nnfm_config;
  nnfm_config.n = 5;
  nnfm_config.learning_rates = {0.01, 0.01, 0.001, 0.001};
  
  for (int i = 0; i < 7; i++) {
    nnfm_models.push_back(std::make_unique<NNFM>(i, nnfm_config));
  }

  // 初始化打印进程中的初始化数据 - 添加NNFM相关数据
  struct {
    std::mutex mutex;  //进程
    bool has_data;
    double time_save;  //定义时间
    franka::RobotState robot_state;
    std::array<double, 7> tau_J_compensate;  //补偿重力和零偏后的测量力矩
    std::array<double, 7> tracking_errors;   // 关节角度跟踪误差
    std::array<double, 7> tau_nnfm_predicted; // NNFM预测的摩擦力矩
    bool use_nnfm; // 是否使用NNFM补偿
  } print_data{};
  std::atomic_bool running{true};
  std::atomic_bool init_csv{false};

  // Start print thread.开始启动打印进程
  std::thread print_thread([print_rate, &print_data, &running, &init_csv, use_nnfm_compensation]() {
    if (!init_csv) {
      //初始化文件csv文件 - 添加NNFM相关列
      std::ofstream outFile;                                     // 创建流对象
      std::string filename = use_nnfm_compensation ? 
                           "test_data_jointmotion_nnfm.csv" : 
                           "test_data_jointmotion_baseline.csv";
      outFile.open(filename, std::ios::out);  //打开文件
      outFile << "time,q1,q2,q3,q4,q5,q6,q7,dq1,dq2,dq3,dq4,dq5,dq6,dq7" << ','
              << "ddq1,ddq2,ddq3,ddq4,ddq5,ddq6,ddq7" << ','
              << "q_d1,q_d2,q_d3,q_d4,q_d5,q_d6,q_d7" << ','  // 期望关节角度
              << "tracking_error1,tracking_error2,tracking_error3,tracking_error4,"  // 跟踪误差
                 "tracking_error5,tracking_error6,tracking_error7" << ','
              << "tau_nnfm1,tau_nnfm2,tau_nnfm3,tau_nnfm4,tau_nnfm5,tau_nnfm6,tau_nnfm7" << ','  // NNFM预测力矩
              << "q1_tau_J_compensate,q2_tau_J_compensate,q3_tau_J_compensate,q4_tau_J_compensate,"
                 "q5_tau_J_compensate,q6_tau_J_compensate,q7_tau_J_compensate" << ','
              << "use_nnfm" << std::endl;  // 是否使用NNFM补偿
      outFile.close();       //关闭文件
      init_csv = true;
    }
    while (running) {
      // 睡眠以实现期望的打印速度
      std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<int>((1.0 / print_rate * 1000.0))));

      // 试图锁住数据，防止读写发生堵塞
      if (print_data.mutex.try_lock()) {
        if (print_data.has_data) {
          // 将结果打印到屏幕上 - 添加NNFM信息显示
          std::cout << "tau_measured_compensate [Nm]: " << print_data.tau_J_compensate << std::endl;
          std::cout << "tracking_errors [rad]: " << print_data.tracking_errors << std::endl;
          if (print_data.use_nnfm) {
            std::cout << "NNFM predicted friction [Nm]: " << print_data.tau_nnfm_predicted << std::endl;
          }
          std::cout << "-----------------------" << std::endl;
          
          // 将结果保存到csv中 - 添加NNFM数据
          if (init_csv) {
            std::ofstream outFile;  // 创建流对象
            std::string filename = print_data.use_nnfm ? 
                                 "test_data_jointmotion_nnfm.csv" : 
                                 "test_data_jointmotion_baseline.csv";
            outFile.open(filename, std::ios::app);
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
                    << print_data.robot_state.q_d[0] << ',' << print_data.robot_state.q_d[1] << ','  // 期望角度
                    << print_data.robot_state.q_d[2] << ',' << print_data.robot_state.q_d[3] << ','
                    << print_data.robot_state.q_d[4] << ',' << print_data.robot_state.q_d[5] << ','
                    << print_data.robot_state.q_d[6] << ','
                    << print_data.tracking_errors[0] << ',' << print_data.tracking_errors[1] << ','  // 跟踪误差
                    << print_data.tracking_errors[2] << ',' << print_data.tracking_errors[3] << ','
                    << print_data.tracking_errors[4] << ',' << print_data.tracking_errors[5] << ','
                    << print_data.tracking_errors[6] << ','
                    << print_data.tau_nnfm_predicted[0] << ',' << print_data.tau_nnfm_predicted[1] << ','  // NNFM预测力矩
                    << print_data.tau_nnfm_predicted[2] << ',' << print_data.tau_nnfm_predicted[3] << ','
                    << print_data.tau_nnfm_predicted[4] << ',' << print_data.tau_nnfm_predicted[5] << ','
                    << print_data.tau_nnfm_predicted[6] << ','
                    << print_data.tau_J_compensate[0] << ',' << print_data.tau_J_compensate[1]
                    << ',' << print_data.tau_J_compensate[2] << ','
                    << print_data.tau_J_compensate[3] << ',' << print_data.tau_J_compensate[4]
                    << ',' << print_data.tau_J_compensate[5] << ','
                    << print_data.tau_J_compensate[6] << ','
                    << (print_data.use_nnfm ? 1 : 0) << std::endl;  // 是否使用NNFM
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
    MotionGenerator motion_generator(0.5, q_goal);  //调用运动生成器
    std::cout << "WARNING: This example will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    robot.control(motion_generator);  //调用运动生成器
    std::cout << "Finished moving to initial joint configuration."
              << std::endl;  //完成初始关节位置配置

    // 永远是在控制回路之前设置附加的参数，而不要在控制回路中设置
    // 设置碰撞检测的行为
    // 设置碰撞的上下限，这里提高了碰撞的阈值，防止力控过程中停止
    robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                               {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

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

    /**下面是自己修改的轨迹部分**/
    auto cartesian_pose_callback = [=, &time](const franka::RobotState&,
                                              franka::Duration period) -> franka::JointVelocities {
      time += period.toSec();  //更新时间

      double cycle = std::floor(std::pow(-1.0, (time - std::fmod(time, time_max)) / time_max));
      double omega = cycle * omega_max / 2.0 * (1.0 - std::cos(2.0 * M_PI / time_max * time));

      franka::JointVelocities velocities = {{0.0, 0.0, 0.0, omega, omega, omega, omega}};

      // 发送理想的位置信息
      if (time >= run_time + 2 * time_max) {
        // running = false;
        std::cout << std::endl << "Finished motion." << std::endl;
        return franka::MotionFinished(velocities);
      }

      return velocities;
    };

    // 设置关节阻抗控制的增益
    // Stiffness 设置刚度
    const std::array<double, 7> k_gains = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
    // Damping  设置阻尼
    const std::array<double, 7> d_gains = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};

    // 定义关节力矩控制回路的回调函数  利用lambda表达式定义回调函数 - 添加NNFM补偿
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback =
            [&time, &print_data, &model, k_gains, d_gains, initial_tau_ext, 
             &nnfm_models, use_nnfm_compensation](
                const franka::RobotState& state, franka::Duration period) -> franka::Torques {
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

      // 计算关节角度跟踪误差
      std::array<double, 7> tracking_errors;
      for (size_t i = 0; i < 7; i++) {
        tracking_errors[i] = state.q_d[i] - state.q[i];
      }

      // NNFM摩擦补偿
      std::array<double, 7> tau_nnfm_predicted = {0};
      if (use_nnfm_compensation) {
        for (size_t i = 0; i < 7; i++) {
          // 使用NNFM预测摩擦力矩
          double friction_pred = nnfm_models[i]->forward(state.dq[i]);
          tau_nnfm_predicted[i] = friction_pred;
          
          // 在线更新NNFM参数
          double measured_friction = tau_ext[i];
          double error = measured_friction - friction_pred;
          nnfm_models[i]->update(state.dq[i], measured_friction, friction_pred, error, 
                               nnfm_models[i]->getModel().optimization_step + 1);
        }
      }

      std::array<double, 7> tau_d_calculated;  //定义计算的控制力矩输出
      for (size_t i = 0; i < 7; i++) {
        double base_torque = k_gains[i] * (state.q_d[i] - state.q[i]) - d_gains[i] * state.dq[i] + coriolis[i];
        
        // 如果使用NNFM补偿，则减去预测的摩擦力矩
        if (use_nnfm_compensation) {
          tau_d_calculated[i] = base_torque - tau_nnfm_predicted[i];
        } else {
          tau_d_calculated[i] = base_torque;
        }
      }

      std::array<double, 7> tau_d_rate_limited =
          franka::limitRate(franka::kMaxTorqueRate, tau_d_calculated, state.tau_J_d);

      //将计算的结果由矩阵形式转化为array
      std::array<double, 7> tau_ext_array{};
      //转回列表形式
      Eigen::VectorXd::Map(&tau_ext_array[0], 7) = tau_ext;

      // Update data to print.更新数据到print结构 - 添加NNFM数据
      if (print_data.mutex.try_lock()) {
        print_data.has_data = true;
        print_data.time_save = time;
        print_data.robot_state = state;
        print_data.tau_J_compensate = tau_ext_array;
        print_data.tracking_errors = tracking_errors;  // 保存跟踪误差
        print_data.tau_nnfm_predicted = tau_nnfm_predicted;  // 保存NNFM预测力矩
        print_data.use_nnfm = use_nnfm_compensation;  // 保存是否使用NNFM
        print_data.mutex.unlock();
      }

      // 发送控制力矩指令
      return tau_d_rate_limited;
    };

    // 开启实时控制回路
    std::cout << "Starting control with " << (use_nnfm_compensation ? "NNFM compensation" : "baseline") << std::endl;
    robot.control(impedance_control_callback, cartesian_pose_callback);

  } catch (const franka::Exception& ex) {
    running = false;
    std::cerr << ex.what() << std::endl;
  }

  if (print_thread.joinable()) {
    print_thread.join();
  }
  
  // 保存NNFM模型参数
  if (use_nnfm_compensation) {
    std::ofstream model_file("nnfm_models_final.txt");
    for (int i = 0; i < 7; i++) {
      auto model = nnfm_models[i]->getModel();
      model_file << "Joint " << i << " NNFM parameters:" << std::endl;
      model_file << "  f_s: " << model.f_s << std::endl;
      model_file << "  w_f: " << model.w_f << std::endl;
      model_file << "  w_odd: ";
      for (double w : model.w_odd) model_file << w << " ";
      model_file << std::endl;
      model_file << "  c_odd: ";
      for (double c : model.c_odd) model_file << c << " ";
      model_file << std::endl;
      model_file << "  optimization_step: " << model.optimization_step << std::endl;
      model_file << "-------------------" << std::endl;
    }
    model_file.close();
    std::cout << "NNFM model parameters saved to nnfm_models_final.txt" << std::endl;
  }
  
  return 0;
}
