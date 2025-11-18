//该程序在圆形阻抗控制的基础上,加入单独控制关节运动的轨迹，用于收集关节的数据.
#include <array>
#include <atomic>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <thread>

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

int main(int argc, char** argv) {
  // Check whether the required arguments were passed.
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }
  const double radius = 0.05;
  //const double vel_max = 0.03;//慢速
  //const double vel_max = 0.06;//中速
  const double vel_max = 0.15;//快速
  const double acceleration_time = 2.0;
  const double run_time = 60.0;

  const double print_rate = 1000;

  double vel_current = 0.0;
  double angle = 0.0;
  double time = 0.0;
  std::array<double, 16> initial_pose{};

  // 初始化打印进程中的初始化数据
  struct {
    std::mutex mutex;  //进程
    bool has_data;
    double time_save;  //定义时间
    franka::RobotState robot_state;
    std::array<double, 7> tau_J_compensate;  //补偿重力和零偏后的测量力矩
  } print_data{};
  std::atomic_bool running{true};
  std::atomic_bool init_csv{false};

  // Start print thread.开始启动打印进程
  std::thread print_thread([print_rate, &print_data, &running, &init_csv]() {
    if (!init_csv) {
      //初始化文件csv文件
      std::ofstream outFile;                                         // 创建流对象
      outFile.open("test_data_yuan_YZ.csv", std::ios::out);  //打开文件
      outFile << "time,q1,q2,q3,q4,q5,q6,q7,dq1,dq2,dq3,dq4,dq5,dq6,dq7" << ','
              << "ddq1,ddq2,ddq3,ddq4,ddq5,ddq6,ddq7" << ','
              << "q1_tau_J_compensate,q2_tau_J_compensate,q3_tau_J_compensate,q4_tau_J_compensate,"
                 "q5_tau_J_compensate,q6_tau_J_compensate,q7_tau_J_compensate"
              << std::endl;  // 初始化第一行
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
          // 将结果打印到屏幕上
          std::cout << "tau_measured_compensate [Nm]: " << print_data.tau_J_compensate << std::endl
                    << "-----------------------" << std::endl;
          // 将结果保存到csv中
          if (init_csv) {
            std::ofstream outFile;  // 创建流对象
            outFile.open("test_data_yuan_YZ.csv", std::ios::app);
            outFile << print_data.time_save << ',' << print_data.robot_state.q[0] << ','
                    << print_data.robot_state.q[1] << ',' << print_data.robot_state.q[2] << ','
                    << print_data.robot_state.q[3] << ',' << print_data.robot_state.q[4] << ','
                    << print_data.robot_state.q[5] << ',' << print_data.robot_state.q[6] << ','
                    << print_data.robot_state.dq[0] << ',' << print_data.robot_state.dq[1] << ','
                    << print_data.robot_state.dq[2] << ',' << print_data.robot_state.dq[3] << ','
                    << print_data.robot_state.dq[4] << ',' << print_data.robot_state.dq[5] << ','
                    << print_data.robot_state.dq[6] << ',' << print_data.robot_state.ddq_d[0] << ','
                    << print_data.robot_state.ddq_d[1] << ',' << print_data.robot_state.ddq_d[2]
                    << ',' << print_data.robot_state.ddq_d[3] << ','
                    << print_data.robot_state.ddq_d[4] << ',' << print_data.robot_state.ddq_d[5]
                    << ',' << print_data.robot_state.ddq_d[6] << ','
                    << print_data.tau_J_compensate[0] << ',' << print_data.tau_J_compensate[1]
                    << ',' << print_data.tau_J_compensate[2] << ','
                    << print_data.tau_J_compensate[3] << ',' << print_data.tau_J_compensate[4]
                    << ',' << print_data.tau_J_compensate[5] << ','
                    << print_data.tau_J_compensate[6] << std::endl;
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

    auto cartesian_pose_callback = [=, &time, &vel_current, &running, &angle, &initial_pose](
                                       const franka::RobotState& robot_state,
                                       franka::Duration period) -> franka::CartesianPose {
      time += period.toSec();

      if (time == 0.0) {
        initial_pose = robot_state.O_T_EE_c;
      }

      if (vel_current < vel_max && time < run_time) {
        vel_current += period.toSec() * std::fabs(vel_max / acceleration_time);
      }

      if (vel_current > 0.0 && time > run_time) {
        vel_current -= period.toSec() * std::fabs(vel_max / acceleration_time);
      }

      vel_current = std::fmax(vel_current, 0.0);

      vel_current = std::fmin(vel_current, vel_max);

      angle += period.toSec() * vel_current / std::fabs(radius);
      if (angle > 2 * M_PI) {
        angle -= 2 * M_PI;
      }

      double delta_y = radius * (1 - std::cos(angle));
      double delta_z = radius * std::sin(angle);

      franka::CartesianPose pose_desired = initial_pose;
      pose_desired.O_T_EE[13] += delta_y;
      pose_desired.O_T_EE[14] += delta_z;

      if (time >= run_time + acceleration_time) {
        running = false;
        return franka::MotionFinished(pose_desired);
      }

      return pose_desired;
    };

    // 设置关节阻抗控制的增益
    // Stiffness 设置刚度
    const std::array<double, 7> k_gains = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
    // Damping  设置阻尼
    const std::array<double, 7> d_gains = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};

    // 定义关节力矩控制回路的回调函数  利用lambda表达式定义回调函数
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback =
            [&time, &print_data, &model, k_gains, d_gains, initial_tau_ext](
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

      // Update data to print.更新数据到print结构
      if (print_data.mutex.try_lock()) {
        print_data.has_data = true;
        print_data.time_save = time;
        print_data.robot_state = state;
        print_data.tau_J_compensate = tau_ext_array;
        print_data.mutex.unlock();
      }

      // 发送控制力矩指令
      return tau_d_rate_limited;
    };

      // 开启实时控制回路
      robot.control(impedance_control_callback, cartesian_pose_callback);

  } catch (const franka::Exception& ex) {
    running = false;
    std::cerr << ex.what() << std::endl;
  }

  if (print_thread.joinable()) {
    print_thread.join();
  }
  return 0;
}
