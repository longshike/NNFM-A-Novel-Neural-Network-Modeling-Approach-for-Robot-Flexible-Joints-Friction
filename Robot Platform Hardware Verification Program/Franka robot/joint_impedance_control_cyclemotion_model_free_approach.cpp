// circular_friction_comparison_control.cpp
// Franka机器人圆形轨迹摩擦模型对比控制程序
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
#include <Eigen/Dense>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include "examples_common.h"

namespace friction_models {

// SVM摩擦模型类
class SVMFrictionModel {
private:
    std::vector<double> SV;  // 支持向量
    std::vector<double> alpha; // 拉格朗日乘子
    double bias;
    double C;
    double epsilon;
    
public:
    SVMFrictionModel() : bias(0), C(1.0), epsilon(0.1) {
        // 初始化预训练参数（简化版本）
        SV = {0.1, 0.5, -0.2, 0.3, -0.1};
        alpha = {0.05, -0.03, 0.02, -0.04, 0.01};
    }
    
    double predict(double velocity) {
        if (SV.empty()) return 0.0;
        
        // 线性核预测
        double prediction = bias;
        for (size_t i = 0; i < SV.size(); i++) {
            prediction += alpha[i] * SV[i] * velocity;
        }
        
        // 添加小量噪声模拟在线学习的不确定性
        double noise_level = 0.01 * std::abs(prediction);
        prediction += noise_level * ((double)rand() / RAND_MAX - 0.5);
        
        return prediction;
    }
    
    void online_update(double velocity, double error) {
        // 简化的在线更新
        if (std::abs(error) > epsilon) {
            // 添加新的支持向量
            SV.push_back(velocity);
            double new_alpha = 0.01 * (error > 0 ? 1.0 : -1.0);
            alpha.push_back(new_alpha);
            
            // 限制支持向量数量
            if (SV.size() > 20) {
                SV.erase(SV.begin());
                alpha.erase(alpha.begin());
            }
            
            // 更新偏置
            bias += 0.001 * (error > 0 ? 1.0 : -1.0);
        }
    }
};

// Light Transformer摩擦模型类
class LightTransformerFrictionModel {
private:
    Eigen::MatrixXd W_xh, W_hh, W_hy;
    Eigen::VectorXd b_h, b_y;
    Eigen::VectorXd h_prev;
    int hidden_units;
    
public:
    LightTransformerFrictionModel(int hidden_size = 6) : hidden_units(hidden_size) {
        // 初始化网络权重
        W_xh = Eigen::MatrixXd::Random(hidden_units, 2) * 0.2;
        W_hh = Eigen::MatrixXd::Random(hidden_units, hidden_units) * 0.2;
        W_hy = Eigen::MatrixXd::Random(1, hidden_units) * 0.2;
        b_h = Eigen::VectorXd::Zero(hidden_units);
        b_y = Eigen::VectorXd::Zero(1);
        h_prev = Eigen::VectorXd::Zero(hidden_units);
    }
    
    double predict(double velocity) {
        // 构建特征向量 [velocity, sign(velocity)]
        Eigen::Vector2d x;
        x << velocity, (velocity > 0 ? 1.0 : (velocity < 0 ? -1.0 : 0.0));
        
        // 前向传播
        Eigen::VectorXd h = (W_xh * x + W_hh * h_prev + b_h).array().tanh();
        double prediction = (W_hy * h + b_y)(0);
        
        // 更新隐藏状态
        h_prev = h;
        
        return prediction;
    }
    
    void online_update(double velocity, double error) {
        // 简化的在线学习更新
        double learning_rate = 0.001;
        
        // 构建特征向量
        Eigen::Vector2d x;
        x << velocity, (velocity > 0 ? 1.0 : (velocity < 0 ? -1.0 : 0.0));
        
        // 前向传播获取隐藏状态
        Eigen::VectorXd h = (W_xh * x + W_hh * h_prev + b_h).array().tanh();
        
        // 计算梯度（简化版本）
        Eigen::VectorXd delta_h = W_hy.transpose() * error;
        delta_h = delta_h.array() * (1 - h.array().square());
        
        // 更新权重
        W_hy += learning_rate * error * h.transpose();
        W_xh += learning_rate * delta_h * x.transpose();
        W_hh += learning_rate * delta_h * h_prev.transpose();
        b_y += learning_rate * error * Eigen::VectorXd::Ones(1);
        b_h += learning_rate * delta_h;
    }
};

// PINN摩擦模型类
class PINNFrictionModel {
private:
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::VectorXd> biases;
    std::vector<int> layer_dims;
    double physics_weight;
    
public:
    PINNFrictionModel() : physics_weight(0.1) {
        // 初始化网络结构 [1, 8, 8, 1]
        layer_dims = {1, 8, 8, 1};
        
        // 初始化权重和偏置
        for (size_t i = 0; i < layer_dims.size() - 1; i++) {
            int input_size = layer_dims[i];
            int output_size = layer_dims[i + 1];
            
            // Xavier初始化
            double scale = std::sqrt(2.0 / (input_size + output_size));
            weights.push_back(Eigen::MatrixXd::Random(output_size, input_size) * scale);
            biases.push_back(Eigen::VectorXd::Zero(output_size));
        }
    }
    
    double predict(double velocity) {
        Eigen::VectorXd activation(1);
        activation(0) = velocity;
        
        // 前向传播
        for (size_t i = 0; i < weights.size(); i++) {
            if (i < weights.size() - 1) {
                // 隐藏层使用tanh激活
                activation = (weights[i] * activation + biases[i]).array().tanh();
            } else {
                // 输出层线性激活
                activation = weights[i] * activation + biases[i];
            }
        }
        
        return activation(0);
    }
    
    void online_update(double velocity, double error) {
        double learning_rate = 0.0001;
        
        // 前向传播记录激活值
        std::vector<Eigen::VectorXd> activations;
        Eigen::VectorXd current_activation(1);
        current_activation(0) = velocity;
        activations.push_back(current_activation);
        
        for (size_t i = 0; i < weights.size(); i++) {
            if (i < weights.size() - 1) {
                current_activation = (weights[i] * current_activation + biases[i]).array().tanh();
            } else {
                current_activation = weights[i] * current_activation + biases[i];
            }
            activations.push_back(current_activation);
        }
        
        // 计算物理约束误差（奇对称性）
        double neg_velocity_prediction = predict(-velocity);
        double physics_error = neg_velocity_prediction + activations.back()(0);
        
        // 反向传播（简化版本）
        Eigen::VectorXd delta = Eigen::VectorXd::Constant(1, -error + physics_weight * physics_error);
        
        for (int i = weights.size() - 1; i >= 0; i--) {
            // 计算梯度
            Eigen::MatrixXd dW = delta * activations[i].transpose();
            Eigen::VectorXd db = delta;
            
            // 更新参数
            weights[i] -= learning_rate * dW;
            biases[i] -= learning_rate * db;
            
            // 传播误差（如果是隐藏层）
            if (i > 0) {
                delta = (weights[i].transpose() * delta).array() * 
                       (1 - activations[i].array().square());
            }
        }
    }
};

// NNFM摩擦模型类
class NNFMFrictionModel {
private:
    double f_s;  // 非光滑部分参数
    double w_f;  // 动态反馈权重
    std::vector<double> w_odd;  // 奇数阶项权重
    std::vector<double> c_odd;  // 奇数阶项系数
    double HH_prev;
    int n;
    
public:
    NNFMFrictionModel(int order = 8) : n(order), HH_prev(0) {
        // 初始化参数
        f_s = 0.15;
        w_f = 0.6;
        
        // 初始化奇数阶项参数
        for (int i = 0; i < n; i++) {
            w_odd.push_back(0.05 + 0.03 * ((double)rand() / RAND_MAX - 0.5));
            c_odd.push_back(0.5 + i * 0.5 + 0.2 * ((double)rand() / RAND_MAX - 0.5));
        }
    }
    
    double predict(double velocity) {
        // 非光滑部分
        double nonsmooth_part = f_s * (velocity > 0 ? 1.0 : (velocity < 0 ? -1.0 : 0.0));
        
        // 光滑部分
        double smooth_part = 0;
        for (int j = 0; j < n; j++) {
            int order = 2 * j + 1;  // 奇数阶
            double linear_component = c_odd[j] * std::pow(velocity, order);
            
            // 组合激活函数
            double tanh_component = std::tanh(linear_component);
            double sigmoid_component = 1.0 / (1.0 + std::exp(-0.5 * linear_component));
            double combined_activation = 0.7 * tanh_component + 0.3 * sigmoid_component;
            
            smooth_part += w_odd[j] * combined_activation;
        }
        
        // 动态反馈
        double HH_current = w_f * HH_prev + smooth_part;
        HH_prev = HH_current;
        
        return HH_current + nonsmooth_part;
    }
    
    void online_update(double velocity, double error) {
        double learning_rate = 0.01;
        
        // 计算梯度（简化版本）
        double grad_f_s = error * (velocity > 0 ? 1.0 : (velocity < 0 ? -1.0 : 0.0));
        
        // 更新参数
        f_s += learning_rate * grad_f_s;
        w_f += learning_rate * error * HH_prev;
        
        // 限制参数范围
        f_s = std::max(0.005, std::min(5.0, f_s));
        w_f = std::max(0.01, std::min(0.99, w_f));
        
        // 更新奇数阶项参数
        for (int j = 0; j < n; j++) {
            int order = 2 * j + 1;
            double linear_component = c_odd[j] * std::pow(velocity, order);
            
            double tanh_component = std::tanh(linear_component);
            double sigmoid_component = 1.0 / (1.0 + std::exp(-0.5 * linear_component));
            double combined_activation = 0.7 * tanh_component + 0.3 * sigmoid_component;
            
            w_odd[j] += learning_rate * error * combined_activation;
            w_odd[j] = std::max(-1.0, std::min(1.0, w_odd[j]));
        }
    }
};

} // namespace friction_models

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
        return -1;
    }

    // 圆形轨迹参数
    const double radius = 0.05;
    const double vel_max = 0.15; // 快速
    const double acceleration_time = 2.0;
    const double run_time = 60.0;
    const double print_rate = 1000.0;

    double vel_current = 0.0;
    double angle = 0.0;
    double time = 0.0;
    std::array<double, 16> initial_pose{};

    // 摩擦模型实例
    friction_models::SVMFrictionModel svm_model;
    friction_models::LightTransformerFrictionModel transformer_model;
    friction_models::PINNFrictionModel pinn_model;
    friction_models::NNFMFrictionModel nnfm_model;

    // 打印数据结构
    struct {
        std::mutex mutex;
        bool has_data;
        double time_save;
        franka::RobotState robot_state;
        std::array<double, 7> tau_J_compensate;
        std::array<double, 7> svm_predictions;
        std::array<double, 7> transformer_predictions;
        std::array<double, 7> pinn_predictions;
        std::array<double, 7> nnfm_predictions;
    } print_data{};
    
    std::atomic_bool running{true};
    std::atomic_bool init_csv{false};

    // 启动打印线程
    std::thread print_thread([print_rate, &print_data, &running, &init_csv]() {
        if (!init_csv) {
            std::ofstream outFile;
            outFile.open("circular_friction_comparison_data.csv", std::ios::out);
            // 扩展CSV头文件，包含所有摩擦模型的预测
            outFile << "time,q1,q2,q3,q4,q5,q6,q7,dq1,dq2,dq3,dq4,dq5,dq6,dq7,"
                    << "ddq1,ddq2,ddq3,ddq4,ddq5,ddq6,ddq7,"
                    << "tau1_compensate,tau2_compensate,tau3_compensate,tau4_compensate,"
                    << "tau5_compensate,tau6_compensate,tau7_compensate,"
                    << "svm1,svm2,svm3,svm4,svm5,svm6,svm7,"
                    << "transformer1,transformer2,transformer3,transformer4,transformer5,transformer6,transformer7,"
                    << "pinn1,pinn2,pinn3,pinn4,pinn5,pinn6,pinn7,"
                    << "nnfm1,nnfm2,nnfm3,nnfm4,nnfm5,nnfm6,nnfm7"
                    << std::endl;
            outFile.close();
            init_csv = true;
        }
        
        while (running) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>((1.0 / print_rate * 1000.0))));

            if (print_data.mutex.try_lock()) {
                if (print_data.has_data) {
                    // 写入扩展的CSV数据
                    std::ofstream outFile;
                    outFile.open("circular_friction_comparison_data.csv", std::ios::app);
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
                            << print_data.tau_J_compensate[6] << ','
                            << print_data.svm_predictions[0] << ',' << print_data.svm_predictions[1] << ','
                            << print_data.svm_predictions[2] << ',' << print_data.svm_predictions[3] << ','
                            << print_data.svm_predictions[4] << ',' << print_data.svm_predictions[5] << ','
                            << print_data.svm_predictions[6] << ','
                            << print_data.transformer_predictions[0] << ',' << print_data.transformer_predictions[1] << ','
                            << print_data.transformer_predictions[2] << ',' << print_data.transformer_predictions[3] << ','
                            << print_data.transformer_predictions[4] << ',' << print_data.transformer_predictions[5] << ','
                            << print_data.transformer_predictions[6] << ','
                            << print_data.pinn_predictions[0] << ',' << print_data.pinn_predictions[1] << ','
                            << print_data.pinn_predictions[2] << ',' << print_data.pinn_predictions[3] << ','
                            << print_data.pinn_predictions[4] << ',' << print_data.pinn_predictions[5] << ','
                            << print_data.pinn_predictions[6] << ','
                            << print_data.nnfm_predictions[0] << ',' << print_data.nnfm_predictions[1] << ','
                            << print_data.nnfm_predictions[2] << ',' << print_data.nnfm_predictions[3] << ','
                            << print_data.nnfm_predictions[4] << ',' << print_data.nnfm_predictions[5] << ','
                            << print_data.nnfm_predictions[6] << std::endl;
                    outFile.close();
                    
                    print_data.has_data = false;
                }
                print_data.mutex.unlock();
            }
        }
    });

    try {
        // 连接机器人
        franka::Robot robot(argv[1]);
        setDefaultBehavior(robot);

        // 移动到初始位置
        std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
        MotionGenerator motion_generator(0.5, q_goal);
        std::cout << "WARNING: This example will move the robot! " << std::endl
                  << "Please make sure to have the user stop button at hand!" << std::endl
                  << "Press Enter to continue..." << std::endl;
        std::cin.ignore();
        robot.control(motion_generator);
        std::cout << "Finished moving to initial joint configuration." << std::endl;

        // 设置碰撞检测
        robot.setCollisionBehavior({{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                                   {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                                   {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
                                   {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

        // 加载模型
        franka::Model model = robot.loadModel();

        // 传感器初始零偏计算
        franka::RobotState initial_state = robot.readOnce();
        Eigen::VectorXd initial_tau_ext(7);
        std::array<double, 7> gravity_array = model.gravity(initial_state);
        Eigen::Map<Eigen::Matrix<double, 7, 1>> initial_tau_measured(initial_state.tau_J.data());
        Eigen::Map<Eigen::Matrix<double, 7, 1>> initial_gravity(gravity_array.data());
        initial_tau_ext = initial_tau_measured - initial_gravity;

        // 圆形轨迹生成回调函数
        auto cartesian_pose_callback = [=, &time, &vel_current, &running, &angle, &initial_pose](
                                           const franka::RobotState& robot_state,
                                           franka::Duration period) -> franka::CartesianPose {
            time += period.toSec();

            if (time == 0.0) {
                initial_pose = robot_state.O_T_EE_c;
            }

            // 速度规划
            if (vel_current < vel_max && time < run_time) {
                vel_current += period.toSec() * std::fabs(vel_max / acceleration_time);
            }

            if (vel_current > 0.0 && time > run_time) {
                vel_current -= period.toSec() * std::fabs(vel_max / acceleration_time);
            }

            vel_current = std::fmax(vel_current, 0.0);
            vel_current = std::fmin(vel_current, vel_max);

            // 角度更新
            angle += period.toSec() * vel_current / std::fabs(radius);
            if (angle > 2 * M_PI) {
                angle -= 2 * M_PI;
            }

            // 计算圆形轨迹
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

        // 关节阻抗控制增益
        const std::array<double, 7> k_gains = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
        const std::array<double, 7> d_gains = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};

        // 阻抗控制回调函数（集成摩擦模型）
        std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
            impedance_control_callback =
                [&, initial_tau_ext](
                    const franka::RobotState& state, franka::Duration period) -> franka::Torques {
                    
                    // 计算补偿扭矩
                    std::array<double, 7> coriolis = model.coriolis(state);
                    std::array<double, 7> gravity_array1 = model.gravity(state);

                    Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis_calculate(coriolis.data());
                    Eigen::Map<const Eigen::Matrix<double, 7, 1>> gravity_calculate(gravity_array1.data());
                    Eigen::Map<const Eigen::Matrix<double, 7, 1>> tau_measured(state.tau_J.data());

                    Eigen::VectorXd tau_ext(7);
                    tau_ext = tau_measured - gravity_calculate - initial_tau_ext;

                    // 计算各摩擦模型的预测
                    std::array<double, 7> svm_predictions{};
                    std::array<double, 7> transformer_predictions{};
                    std::array<double, 7> pinn_predictions{};
                    std::array<double, 7> nnfm_predictions{};

                    for (size_t i = 0; i < 7; i++) {
                        double velocity = state.dq[i];
                        
                        // 各模型预测
                        svm_predictions[i] = svm_model.predict(velocity);
                        transformer_predictions[i] = transformer_model.predict(velocity);
                        pinn_predictions[i] = pinn_model.predict(velocity);
                        nnfm_predictions[i] = nnfm_model.predict(velocity);
                        
                        // 在线更新（可选）
                        double measured_friction = tau_ext(i);
                        double prediction_error = measured_friction - svm_predictions[i];
                        
                        // 根据需要进行在线学习
                        static int update_counter = 0;
                        if (update_counter++ % 10 == 0) { // 每10步更新一次，减少计算负载
                            svm_model.online_update(velocity, prediction_error);
                            transformer_model.online_update(velocity, prediction_error);
                            pinn_model.online_update(velocity, prediction_error);
                            nnfm_model.online_update(velocity, prediction_error);
                        }
                    }

                    // 计算控制力矩
                    std::array<double, 7> tau_d_calculated;
                    for (size_t i = 0; i < 7; i++) {
                        // 基础阻抗控制
                        tau_d_calculated[i] =
                            k_gains[i] * (state.q_d[i] - state.q[i]) - d_gains[i] * state.dq[i] + coriolis[i];
                        
                        // 可选：添加摩擦补偿（取消注释以启用）
                        // double avg_friction_compensation = (svm_predictions[i] + 
                        //                                   transformer_predictions[i] + 
                        //                                   pinn_predictions[i] + 
                        //                                   nnfm_predictions[i]) / 4.0;
                        // tau_d_calculated[i] += avg_friction_compensation;
                    }

                    std::array<double, 7> tau_d_rate_limited =
                        franka::limitRate(franka::kMaxTorqueRate, tau_d_calculated, state.tau_J_d);

                    // 转换回数组形式
                    std::array<double, 7> tau_ext_array{};
                    Eigen::VectorXd::Map(&tau_ext_array[0], 7) = tau_ext;

                    // 更新打印数据
                    if (print_data.mutex.try_lock()) {
                        print_data.has_data = true;
                        print_data.time_save = time;
                        print_data.robot_state = state;
                        print_data.tau_J_compensate = tau_ext_array;
                        print_data.svm_predictions = svm_predictions;
                        print_data.transformer_predictions = transformer_predictions;
                        print_data.pinn_predictions = pinn_predictions;
                        print_data.nnfm_predictions = nnfm_predictions;
                        print_data.mutex.unlock();
                    }

                    return tau_d_rate_limited;
                };

        // 开始控制
        std::cout << "Starting circular friction comparison control..." << std::endl;
        std::cout << "Circular trajectory parameters:" << std::endl;
        std::cout << "  Radius: " << radius << " m" << std::endl;
        std::cout << "  Max velocity: " << vel_max << " m/s" << std::endl;
        std::cout << "  Run time: " << run_time << " s" << std::endl;
        
        robot.control(impedance_control_callback, cartesian_pose_callback);

    } catch (const franka::Exception& ex) {
        running = false;
        std::cerr << ex.what() << std::endl;
    }

    if (print_thread.joinable()) {
        print_thread.join();
    }
    
    std::cout << "Circular friction comparison experiment completed." << std::endl;
    std::cout << "Data saved to circular_friction_comparison_data.csv" << std::endl;
    return 0;
}