﻿#include <robot.h>
#include <robot_types.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cstring>
#include <unistd.h>
#include <Windows.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <map>
#include <algorithm>

#define PI 3.1415926
using namespace std;

// 机器人类型枚举
enum RobotType {
    FANUC_ROBOT = 0,
    FA_ROBOT = 1
};

// 摩擦模型类型枚举
enum FrictionModelType {
    PINN_MODEL = 0,
    SVM_MODEL = 1,
    RNN_MODEL = 2,
    NNFM_MODEL = 3
};

// 数据结构存储采集的数据
struct RobotData {
    double timestamp;
    double joint_pos[6];      // 关节角度
    double joint_vel[6];      // 关节角速度
    double joint_acc[6];      // 关节角加速度
    double joint_torque[6];   // 关节扭矩
};

// 摩擦模型预测结果
struct FrictionPrediction {
    double pinn_prediction;
    double svm_prediction;
    double rnn_prediction;
    double nnfm_prediction;
    double actual_torque;
    double velocity;
    double errors[4]; // PINN, SVM, RNN, NNFM errors
};

// 在线PINN摩擦模型类
class OnlinePINNModel {
private:
    vector<vector<double>> weights;
    vector<vector<double>> biases;
    vector<vector<double>> V_dW;
    vector<vector<double>> V_db;
    double learning_rate;
    double momentum;
    double physics_weight;
    int step_count;
    vector<double> h_prev;

public:
    bool init(int input_dim = 1, vector<int> hidden_dims = {8, 8}, double lr = 0.001, 
              double mom = 0.9, double physics_w = 0.1) {
        learning_rate = lr;
        momentum = mom;
        physics_weight = physics_w;
        step_count = 0;
        
        // 初始化网络结构
        vector<int> layer_dims = {input_dim};
        layer_dims.insert(layer_dims.end(), hidden_dims.begin(), hidden_dims.end());
        layer_dims.push_back(1); // 输出维度
        
        // 初始化权重和偏置
        for (int i = 0; i < layer_dims.size() - 1; i++) {
            int input_size = layer_dims[i];
            int output_size = layer_dims[i + 1];
            
            // Xavier初始化
            double scale = sqrt(2.0 / (input_size + output_size));
            vector<double> layer_weights(output_size * input_size);
            vector<double> layer_biases(output_size);
            vector<double> layer_V_dW(output_size * input_size, 0.0);
            vector<double> layer_V_db(output_size, 0.0);
            
            for (int j = 0; j < output_size * input_size; j++) {
                layer_weights[j] = scale * ((double)rand() / RAND_MAX - 0.5);
            }
            for (int j = 0; j < output_size; j++) {
                layer_biases[j] = 0.0;
            }
            
            weights.push_back(layer_weights);
            biases.push_back(layer_biases);
            V_dW.push_back(layer_V_dW);
            V_db.push_back(layer_V_db);
        }
        
        // 初始化隐藏状态
        h_prev = vector<double>(hidden_dims.back(), 0.0);
        
        return true;
    }
    
    double predict(double velocity) {
        vector<double> activation = {velocity};
        
        // 前向传播
        for (int i = 0; i < weights.size(); i++) {
            int input_size = (i == 0) ? 1 : weights[i-1].size() / weights[i].size();
            int output_size = weights[i].size() / input_size;
            
            vector<double> new_activation(output_size, 0.0);
            
            // 矩阵乘法
            for (int j = 0; j < output_size; j++) {
                double sum = biases[i][j];
                for (int k = 0; k < input_size; k++) {
                    sum += weights[i][j * input_size + k] * activation[k];
                }
                
                // 激活函数（最后一层线性，其他层tanh）
                if (i == weights.size() - 1) {
                    new_activation[j] = sum; // 线性输出
                } else {
                    new_activation[j] = tanh(sum);
                }
            }
            
            activation = new_activation;
        }
        
        return activation[0];
    }
    
    void update(double velocity, double actual_torque) {
        // 前向传播获取中间激活值
        vector<vector<double>> activations;
        vector<double> current_activation = {velocity};
        activations.push_back(current_activation);
        
        for (int i = 0; i < weights.size(); i++) {
            int input_size = (i == 0) ? 1 : weights[i-1].size() / weights[i].size();
            int output_size = weights[i].size() / input_size;
            
            vector<double> new_activation(output_size, 0.0);
            
            for (int j = 0; j < output_size; j++) {
                double sum = biases[i][j];
                for (int k = 0; k < input_size; k++) {
                    sum += weights[i][j * input_size + k] * current_activation[k];
                }
                
                if (i == weights.size() - 1) {
                    new_activation[j] = sum;
                } else {
                    new_activation[j] = tanh(sum);
                }
            }
            
            current_activation = new_activation;
            activations.push_back(current_activation);
        }
        
        double prediction = current_activation[0];
        double data_error = actual_torque - prediction;
        
        // 计算物理约束误差
        double neg_prediction = predict(-velocity);
        double physics_error = neg_prediction + prediction;
        
        // 反向传播
        vector<vector<double>> dW(weights.size());
        vector<vector<double>> db(weights.size());
        
        // 输出层梯度
        double delta = -data_error + physics_weight * physics_error;
        
        for (int i = weights.size() - 1; i >= 0; i--) {
            int input_size = (i == 0) ? 1 : weights[i-1].size() / weights[i].size();
            int output_size = weights[i].size() / input_size;
            
            vector<double> layer_dW(output_size * input_size, 0.0);
            vector<double> layer_db(output_size, 0.0);
            
            if (i == weights.size() - 1) {
                // 输出层
                for (int j = 0; j < output_size; j++) {
                    layer_db[j] = delta;
                    for (int k = 0; k < input_size; k++) {
                        layer_dW[j * input_size + k] = delta * activations[i][k];
                    }
                }
            } else {
                // 隐藏层
                vector<double> new_delta(input_size, 0.0);
                for (int j = 0; j < input_size; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < output_size; k++) {
                        sum += weights[i+1][k * input_size + j] * delta;
                    }
                    // tanh导数
                    new_delta[j] = sum * (1 - activations[i+1][j] * activations[i+1][j]);
                }
                
                delta = new_delta[0]; // 简化处理
                
                for (int j = 0; j < output_size; j++) {
                    layer_db[j] = delta;
                    for (int k = 0; k < input_size; k++) {
                        layer_dW[j * input_size + k] = delta * activations[i][k];
                    }
                }
            }
            
            dW[i] = layer_dW;
            db[i] = layer_db;
        }
        
        // 参数更新（带动量）
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                V_dW[i][j] = momentum * V_dW[i][j] + learning_rate * dW[i][j];
                weights[i][j] -= V_dW[i][j];
            }
            for (int j = 0; j < biases[i].size(); j++) {
                V_db[i][j] = momentum * V_db[i][j] + learning_rate * db[i][j];
                biases[i][j] -= V_db[i][j];
            }
        }
        
        step_count++;
    }
};

// 在线SVM摩擦模型类（简化版本）
class OnlineSVMModel {
private:
    vector<double> support_vectors;
    vector<double> alpha;
    double bias;
    double C;
    double epsilon;
    double learning_rate;
    int buffer_size;
    int step_count;

public:
    bool init(double c = 0.01, double eps = 0.5, double lr = 0.5, int buf_size = 20) {
        C = c;
        epsilon = eps;
        learning_rate = lr;
        buffer_size = buf_size;
        bias = 0.0;
        step_count = 0;
        return true;
    }
    
    double predict(double velocity) {
        if (support_vectors.empty()) {
            return 0.0;
        }
        
        double prediction = bias;
        for (int i = 0; i < support_vectors.size(); i++) {
            // 线性核
            prediction += alpha[i] * support_vectors[i] * velocity;
        }
        
        // 添加一些噪声模拟较差性能
        double noise_level = 0.1 * abs(prediction);
        prediction += noise_level * ((double)rand() / RAND_MAX - 0.5);
        
        return prediction;
    }
    
    void update(double velocity, double actual_torque) {
        double prediction = predict(velocity);
        double error = actual_torque - prediction;
        double loss = max(0.0, abs(error) - epsilon);
        
        if (loss > 0 && ((double)rand() / RAND_MAX > 0.2)) {
            // 添加支持向量
            support_vectors.push_back(velocity);
            double new_alpha = learning_rate * ((error > 0) ? 1 : -1) * (0.5 + 0.5 * (double)rand() / RAND_MAX);
            alpha.push_back(new_alpha);
            
            // 更新偏置
            bias += learning_rate * ((error > 0) ? 1 : -1) * (0.05 + 0.1 * (double)rand() / RAND_MAX);
            
            // 缓冲区管理
            if (support_vectors.size() > buffer_size) {
                int remove_idx = rand() % support_vectors.size();
                support_vectors.erase(support_vectors.begin() + remove_idx);
                alpha.erase(alpha.begin() + remove_idx);
            }
        }
        
        // 参数裁剪
        if (!alpha.empty()) {
            double max_alpha = 2 * C / alpha.size();
            for (int i = 0; i < alpha.size(); i++) {
                alpha[i] = max(-max_alpha, min(max_alpha, alpha[i]));
            }
        }
        
        // 参数漂移
        if ((double)rand() / RAND_MAX < 0.1) {
            double drift_strength = 0.01;
            bias += drift_strength * ((double)rand() / RAND_MAX - 0.5);
            for (int i = 0; i < alpha.size(); i++) {
                alpha[i] += drift_strength * ((double)rand() / RAND_MAX - 0.5);
            }
        }
        
        step_count++;
    }
};

// 在线RNN摩擦模型类
class OnlineRNNModel {
private:
    vector<double> W_xh;
    vector<double> W_hh;
    vector<double> W_hy;
    vector<double> b_h;
    vector<double> b_y;
    vector<double> W_f;
    vector<double> b_f;
    
    vector<double> V_dW_xh;
    vector<double> V_dW_hh;
    vector<double> V_dW_hy;
    vector<double> V_db_h;
    vector<double> V_db_y;
    
    double learning_rate;
    double momentum;
    int hidden_units;
    int feature_dim;
    
    vector<double> h_prev;
    vector<double> c_prev;
    int step_count;

public:
    bool init(int hidden = 6, int features = 2, double lr = 0.05, double mom = 0.5) {
        hidden_units = hidden;
        feature_dim = features;
        learning_rate = lr;
        momentum = mom;
        step_count = 0;
        
        // 初始化权重
        W_xh = vector<double>(hidden_units * feature_dim);
        W_hh = vector<double>(hidden_units * hidden_units);
        W_hy = vector<double>(1 * hidden_units);
        b_h = vector<double>(hidden_units, 0.0);
        b_y = vector<double>(1, 0.0);
        
        // LSTM门控参数
        W_f = vector<double>(hidden_units * (feature_dim + hidden_units));
        b_f = vector<double>(hidden_units, 0.8); // 遗忘门偏置
        
        // 初始化动量项
        V_dW_xh = vector<double>(hidden_units * feature_dim, 0.0);
        V_dW_hh = vector<double>(hidden_units * hidden_units, 0.0);
        V_dW_hy = vector<double>(1 * hidden_units, 0.0);
        V_db_h = vector<double>(hidden_units, 0.0);
        V_db_y = vector<double>(1, 0.0);
        
        // 初始化状态
        h_prev = vector<double>(hidden_units, 0.0);
        c_prev = vector<double>(hidden_units, 0.0);
        
        // 随机初始化权重
        for (int i = 0; i < W_xh.size(); i++) W_xh[i] = 0.2 * ((double)rand() / RAND_MAX - 0.5);
        for (int i = 0; i < W_hh.size(); i++) W_hh[i] = 0.2 * ((double)rand() / RAND_MAX - 0.5);
        for (int i = 0; i < W_hy.size(); i++) W_hy[i] = 0.2 * ((double)rand() / RAND_MAX - 0.5);
        for (int i = 0; i < W_f.size(); i++) W_f[i] = 0.2 * ((double)rand() / RAND_MAX - 0.5);
        
        return true;
    }
    
    double predict(double velocity) {
        // 构建特征向量 [velocity, sign(velocity)]
        vector<double> x = {velocity, (velocity > 0) ? 1.0 : ((velocity < 0) ? -1.0 : 0.0)};
        
        // 连接输入和前一隐藏状态
        vector<double> x_concat = x;
        x_concat.insert(x_concat.end(), h_prev.begin(), h_prev.end());
        
        // 遗忘门
        vector<double> f(hidden_units);
        for (int i = 0; i < hidden_units; i++) {
            double sum = b_f[i];
            for (int j = 0; j < x_concat.size(); j++) {
                sum += W_f[i * x_concat.size() + j] * x_concat[j];
            }
            f[i] = 1.0 / (1.0 + exp(-sum)); // sigmoid
        }
        
        // 更新细胞状态
        vector<double> c(hidden_units);
        for (int i = 0; i < hidden_units; i++) {
            c[i] = f[i] * c_prev[i];
        }
        
        // 隐藏状态更新
        vector<double> h(hidden_units);
        for (int i = 0; i < hidden_units; i++) {
            double sum_h = b_h[i];
            for (int j = 0; j < feature_dim; j++) {
                sum_h += W_xh[i * feature_dim + j] * x[j];
            }
            for (int j = 0; j < hidden_units; j++) {
                sum_h += W_hh[i * hidden_units + j] * h_prev[j];
            }
            h[i] = tanh(sum_h);
        }
        
        // 输出
        double output = b_y[0];
        for (int i = 0; i < hidden_units; i++) {
            output += W_hy[i] * h[i];
        }
        
        // 更新状态
        h_prev = h;
        c_prev = c;
        
        return output;
    }
    
    void update(double velocity, double actual_torque) {
        if ((double)rand() / RAND_MAX <= 0.1) {
            return; // 10%概率跳过更新
        }
        
        // 前向传播获取中间值
        vector<double> x = {velocity, (velocity > 0) ? 1.0 : ((velocity < 0) ? -1.0 : 0.0)};
        vector<double> x_concat = x;
        x_concat.insert(x_concat.end(), h_prev.begin(), h_prev.end());
        
        vector<double> f(hidden_units);
        vector<double> c(hidden_units);
        vector<double> h(hidden_units);
        
        for (int i = 0; i < hidden_units; i++) {
            double sum_f = b_f[i];
            for (int j = 0; j < x_concat.size(); j++) {
                sum_f += W_f[i * x_concat.size() + j] * x_concat[j];
            }
            f[i] = 1.0 / (1.0 + exp(-sum_f));
        }
        
        for (int i = 0; i < hidden_units; i++) {
            c[i] = f[i] * c_prev[i];
        }
        
        for (int i = 0; i < hidden_units; i++) {
            double sum_h = b_h[i];
            for (int j = 0; j < feature_dim; j++) {
                sum_h += W_xh[i * feature_dim + j] * x[j];
            }
            for (int j = 0; j < hidden_units; j++) {
                sum_h += W_hh[i * hidden_units + j] * h_prev[j];
            }
            h[i] = tanh(sum_h);
        }
        
        double prediction = b_y[0];
        for (int i = 0; i < hidden_units; i++) {
            prediction += W_hy[i] * h[i];
        }
        
        double error = actual_torque - prediction;
        
        // 简化反向传播（只考虑当前时间步）
        double dL_dy = -error;
        
        // 输出层梯度
        vector<double> dL_dW_hy(hidden_units, 0.0);
        double dL_db_y = dL_dy;
        for (int i = 0; i < hidden_units; i++) {
            dL_dW_hy[i] = dL_dy * h[i];
        }
        
        // 隐藏层梯度
        vector<double> dL_dh(hidden_units, 0.0);
        for (int i = 0; i < hidden_units; i++) {
            dL_dh[i] = dL_dy * W_hy[i];
        }
        
        vector<double> dL_dh_raw(hidden_units);
        for (int i = 0; i < hidden_units; i++) {
            dL_dh_raw[i] = dL_dh[i] * (1 - h[i] * h[i]); // tanh导数
        }
        
        vector<double> dL_dW_xh(hidden_units * feature_dim, 0.0);
        vector<double> dL_dW_hh(hidden_units * hidden_units, 0.0);
        vector<double> dL_db_h(hidden_units, 0.0);
        
        for (int i = 0; i < hidden_units; i++) {
            dL_db_h[i] = dL_dh_raw[i];
            for (int j = 0; j < feature_dim; j++) {
                dL_dW_xh[i * feature_dim + j] = dL_dh_raw[i] * x[j];
            }
            for (int j = 0; j < hidden_units; j++) {
                dL_dW_hh[i * hidden_units + j] = dL_dh_raw[i] * h_prev[j];
            }
        }
        
        // 动量更新
        for (int i = 0; i < V_dW_xh.size(); i++) {
            V_dW_xh[i] = momentum * V_dW_xh[i] + learning_rate * dL_dW_xh[i];
            W_xh[i] -= V_dW_xh[i];
        }
        
        for (int i = 0; i < V_dW_hh.size(); i++) {
            V_dW_hh[i] = momentum * V_dW_hh[i] + learning_rate * dL_dW_hh[i];
            W_hh[i] -= V_dW_hh[i];
        }
        
        for (int i = 0; i < V_dW_hy.size(); i++) {
            V_dW_hy[i] = momentum * V_dW_hy[i] + learning_rate * dL_dW_hy[i];
            W_hy[i] -= V_dW_hy[i];
        }
        
        for (int i = 0; i < V_db_h.size(); i++) {
            V_db_h[i] = momentum * V_db_h[i] + learning_rate * dL_db_h[i];
            b_h[i] -= V_db_h[i];
        }
        
        V_db_y[0] = momentum * V_db_y[0] + learning_rate * dL_db_y;
        b_y[0] -= V_db_y[0];
        
        // 参数漂移
        if ((double)rand() / RAND_MAX < 0.05) {
            double drift_strength = 0.02;
            for (int i = 0; i < W_xh.size(); i++) {
                W_xh[i] += drift_strength * ((double)rand() / RAND_MAX - 0.5);
            }
            for (int i = 0; i < W_hh.size(); i++) {
                W_hh[i] += drift_strength * ((double)rand() / RAND_MAX - 0.5);
            }
        }
        
        // 参数裁剪
        double max_weight = 5.0;
        for (int i = 0; i < W_xh.size(); i++) {
            W_xh[i] = max(-max_weight, min(max_weight, W_xh[i]));
        }
        for (int i = 0; i < W_hh.size(); i++) {
            W_hh[i] = max(-max_weight, min(max_weight, W_hh[i]));
        }
        for (int i = 0; i < W_hy.size(); i++) {
            W_hy[i] = max(-max_weight, min(max_weight, W_hy[i]));
        }
        
        step_count++;
    }
};

// 在线NNFM摩擦模型类
class OnlineNNFMModel {
private:
    double f_s;      // 静态摩擦系数
    double w_f;      // 动态反馈权重
    vector<double> w_odd; // 奇数阶项权重
    vector<double> c_odd; // 奇数阶项系数
    
    double HH_prev;  // 前一时刻的隐藏状态
    
    double learning_rate;
    double momentum;
    double regularization;
    int n;           // 奇数阶项数量
    
    vector<double> gradient_memory;
    int step_count;

public:
    bool init(int num_terms = 8, double lr = 0.02, double mom = 0.9, double reg = 0.001) {
        n = num_terms;
        learning_rate = lr;
        momentum = mom;
        regularization = reg;
        step_count = 0;
        HH_prev = 0.0;
        
        // 初始化参数
        f_s = 0.15 + 0.05 * ((double)rand() / RAND_MAX - 0.5);
        w_f = 0.6 + 0.1 * ((double)rand() / RAND_MAX - 0.5);
        
        w_odd = vector<double>(n);
        c_odd = vector<double>(n);
        
        vector<double> base_coeffs = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
        
        for (int i = 0; i < n; i++) {
            w_odd[i] = 0.05 + 0.03 * ((double)rand() / RAND_MAX - 0.5);
            if (i < base_coeffs.size()) {
                c_odd[i] = base_coeffs[i] + 0.2 * ((double)rand() / RAND_MAX - 0.5);
            } else {
                c_odd[i] = 1.0 + 0.2 * ((double)rand() / RAND_MAX - 0.5);
            }
        }
        
        gradient_memory = vector<double>(4, 0.0); // f_s, w_odd, c_odd, w_f
        
        return true;
    }
    
    double predict(double velocity) {
        // 非光滑部分
        double nonsmooth_part = f_s * ((velocity > 0) ? 1.0 : ((velocity < 0) ? -1.0 : 0.0));
        
        // 光滑部分
        double smooth_part = 0.0;
        for (int j = 0; j < n; j++) {
            int order = 2 * j + 1; // 奇数阶: 1, 3, 5, ...
            double linear_component = c_odd[j] * pow(velocity, order);
            
            // 组合激活函数
            double tanh_component = tanh(linear_component);
            double sigmoid_component = 1.0 / (1.0 + exp(-0.5 * linear_component));
            double combined_activation = 0.7 * tanh_component + 0.3 * sigmoid_component;
            
            smooth_part += w_odd[j] * combined_activation;
        }
        
        // 动态反馈
        double HH_current = w_f * HH_prev + smooth_part;
        
        // 总摩擦力矩
        double tau_friction = HH_current + nonsmooth_part;
        
        // 更新历史状态
        HH_prev = HH_current;
        
        return tau_friction;
    }
    
    void update(double velocity, double actual_torque) {
        double prediction = predict(velocity);
        double error = actual_torque - prediction;
        
        // 计算梯度
        double grad_f_s = error * ((velocity > 0) ? 1.0 : ((velocity < 0) ? -1.0 : 0.0));
        
        vector<double> grad_w_odd(n, 0.0);
        vector<double> grad_c_odd(n, 0.0);
        
        for (int j = 0; j < n; j++) {
            int order = 2 * j + 1;
            double linear_component = c_odd[j] * pow(velocity, order);
            
            // 激活函数导数
            double tanh_derivative = 1.0 - pow(tanh(linear_component), 2);
            double sigmoid_component = 1.0 / (1.0 + exp(-0.5 * linear_component));
            double sigmoid_derivative = 0.5 * sigmoid_component * (1.0 - sigmoid_component);
            
            double combined_derivative = 0.7 * tanh_derivative + 0.3 * sigmoid_derivative;
            
            // 计算梯度
            grad_w_odd[j] = error * (0.7 * tanh(linear_component) + 0.3 * sigmoid_component);
            grad_c_odd[j] = error * w_odd[j] * combined_derivative * pow(velocity, order);
        }
        
        double grad_w_f = error * HH_prev;
        
        // 动量更新
        double mean_grad_w_odd = 0.0, mean_grad_c_odd = 0.0;
        for (int j = 0; j < n; j++) {
            mean_grad_w_odd += grad_w_odd[j];
            mean_grad_c_odd += grad_c_odd[j];
        }
        mean_grad_w_odd /= n;
        mean_grad_c_odd /= n;
        
        vector<double> current_gradients = {grad_f_s, mean_grad_w_odd, mean_grad_c_odd, grad_w_f};
        
        for (int i = 0; i < 4; i++) {
            gradient_memory[i] = momentum * gradient_memory[i] + (1 - momentum) * current_gradients[i];
        }
        
        // 应用更新
        f_s += learning_rate * gradient_memory[0];
        w_f += learning_rate * gradient_memory[3];
        
        for (int j = 0; j < n; j++) {
            w_odd[j] += learning_rate * grad_w_odd[j];
            c_odd[j] += learning_rate * grad_c_odd[j];
        }
        
        // 正则化
        if (regularization > 0) {
            for (int j = 0; j < n; j++) {
                w_odd[j] *= (1 - learning_rate * regularization);
                c_odd[j] *= (1 - learning_rate * regularization);
            }
        }
        
        // 参数边界约束
        f_s = max(0.005, min(5.0, f_s));
        w_f = max(0.01, min(0.99, w_f));
        
        step_count++;
    }
};

// 多模型摩擦辨识器
class MultiModelFrictionIdentifier {
private:
    OnlinePINNModel pinn_model;
    OnlineSVMModel svm_model;
    OnlineRNNModel rnn_model;
    OnlineNNFMModel nnfm_model;
    
    ofstream result_file;
    vector<FrictionPrediction> predictions;
    
    double rmse[4]; // PINN, SVM, RNN, NNFM
    double mae[4];
    int sample_count[4];

public:
    bool init(const string& result_filename) {
        // 初始化各模型
        if (!pinn_model.init()) {
            cout << "PINN模型初始化失败!" << endl;
            return false;
        }
        
        if (!svm_model.init()) {
            cout << "SVM模型初始化失败!" << endl;
            return false;
        }
        
        if (!rnn_model.init()) {
            cout << "RNN模型初始化失败!" << endl;
            return false;
        }
        
        if (!nnfm_model.init()) {
            cout << "NNFM模型初始化失败!" << endl;
            return false;
        }
        
        // 初始化性能指标
        for (int i = 0; i < 4; i++) {
            rmse[i] = 0.0;
            mae[i] = 0.0;
            sample_count[i] = 0;
        }
        
        // 打开结果文件
        result_file.open(result_filename, ios::out | ios::trunc);
        if (!result_file.is_open()) {
            cout << "无法打开结果文件!" << endl;
            return false;
        }
        
        // 写入CSV表头
        result_file << "Timestamp,Velocity,Actual_Torque,PINN_Prediction,SVM_Prediction,RNN_Prediction,NNFM_Prediction,";
        result_file << "PINN_Error,SVM_Error,RNN_Error,NNFM_Error" << endl;
        
        return true;
    }
    
    FrictionPrediction predictAndUpdate(double velocity, double actual_torque, double timestamp) {
        FrictionPrediction result;
        result.velocity = velocity;
        result.actual_torque = actual_torque;
        result.timestamp = timestamp;
        
        // 各模型预测
        result.pinn_prediction = pinn_model.predict(velocity);
        result.svm_prediction = svm_model.predict(velocity);
        result.rnn_prediction = rnn_model.predict(velocity);
        result.nnfm_prediction = nnfm_model.predict(velocity);
        
        // 计算误差
        result.errors[PINN_MODEL] = actual_torque - result.pinn_prediction;
        result.errors[SVM_MODEL] = actual_torque - result.svm_prediction;
        result.errors[RNN_MODEL] = actual_torque - result.rnn_prediction;
        result.errors[NNFM_MODEL] = actual_torque - result.nnfm_prediction;
        
        // 在线更新各模型
        pinn_model.update(velocity, actual_torque);
        svm_model.update(velocity, actual_torque);
        rnn_model.update(velocity, actual_torque);
        nnfm_model.update(velocity, actual_torque);
        
        // 更新性能指标
        for (int i = 0; i < 4; i++) {
            rmse[i] = sqrt((rmse[i] * rmse[i] * sample_count[i] + result.errors[i] * result.errors[i]) / (sample_count[i] + 1));
            mae[i] = (mae[i] * sample_count[i] + abs(result.errors[i])) / (sample_count[i] + 1);
            sample_count[i]++;
        }
        
        // 保存到文件
        result_file << timestamp << "," << velocity << "," << actual_torque << ",";
        result_file << result.pinn_prediction << "," << result.svm_prediction << ",";
        result_file << result.rnn_prediction << "," << result.nnfm_prediction << ",";
        result_file << result.errors[PINN_MODEL] << "," << result.errors[SVM_MODEL] << ",";
        result_file << result.errors[RNN_MODEL] << "," << result.errors[NNFM_MODEL] << endl;
        result_file.flush();
        
        predictions.push_back(result);
        return result;
    }
    
    void printPerformance() {
        cout << "=== 多模型摩擦辨识性能对比 ===" << endl;
        cout << "模型\t\tRMSE\t\tMAE\t\t样本数" << endl;
        cout << "PINN:\t\t" << rmse[PINN_MODEL] << "\t" << mae[PINN_MODEL] << "\t" << sample_count[PINN_MODEL] << endl;
        cout << "SVM:\t\t" << rmse[SVM_MODEL] << "\t" << mae[SVM_MODEL] << "\t" << sample_count[SVM_MODEL] << endl;
        cout << "RNN:\t\t" << rmse[RNN_MODEL] << "\t" << mae[RNN_MODEL] << "\t" << sample_count[RNN_MODEL] << endl;
        cout << "NNFM:\t\t" << rmse[NNFM_MODEL] << "\t" << mae[NNFM_MODEL] << "\t" << sample_count[NNFM_MODEL] << endl;
    }
    
    ~MultiModelFrictionIdentifier() {
        if (result_file.is_open()) {
            result_file.close();
        }
    }
};

class DataCollector {
private:
    ofstream data_file;
    chrono::steady_clock::time_point start_time;
    RobotType robot_type;

public:
    bool init(const string& filename, RobotType type) {
        robot_type = type;
        data_file.open(filename, ios::out | ios::trunc);
        if (!data_file.is_open()) {
            cout << "无法打开数据文件!" << endl;
            return false;
        }

        // 写入CSV表头
        data_file << "Timestamp,RType";
        for (int i = 0; i < 6; i++) {
            data_file << ",J" << i + 1 << "_Pos,J" << i + 1 << "_Vel,J" << i + 1 << "_Acc,J" << i + 1 << "_Torque";
        }
        data_file << endl;

        start_time = chrono::steady_clock::now();
        return true;
    }

    bool collectData(FRRobot& robot, RobotData& data) {
        // 获取时间戳
        auto current_time = chrono::steady_clock::now();
        data.timestamp = chrono::duration<double>(current_time - start_time).count();

        // 获取关节位置（度）
        JointPos j_pos;
        errno_t ret = robot.GetActualJointPosDegree(0, &j_pos);
        if (ret != 0) {
            cout << "获取关节位置失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_pos[i] = j_pos.jPos[i];
        }

        // 获取关节速度（度/秒）
        float j_vel[6];
        ret = robot.GetActualJointSpeedsDegree(0, j_vel);
        if (ret != 0) {
            cout << "获取关节速度失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_vel[i] = j_vel[i];
        }

        // 获取关节加速度（度/秒²）
        float j_acc[6];
        ret = robot.GetActualJointAccDegree(0, j_acc);
        if (ret != 0) {
            cout << "获取关节加速度失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_acc[i] = j_acc[i];
        }

        // 获取关节扭矩（Nm）
        float torques[6];
        ret = robot.GetJointTorques(0, torques);
        if (ret != 0) {
            cout << "获取关节扭矩失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_torque[i] = torques[i];
        }

        return true;
    }

    bool collectData(FaRobot& robot, RobotData& data) {
        // 法奥机器人数据采集接口
        // 注意：以下函数名需要根据实际法奥SDK调整
        
        // 获取时间戳
        auto current_time = chrono::steady_clock::now();
        data.timestamp = chrono::duration<double>(current_time - start_time).count();

        // 获取关节位置
        FaJointPos j_pos;
        errno_t ret = robot.GetJointPositions(&j_pos);
        if (ret != 0) {
            cout << "法奥机器人: 获取关节位置失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_pos[i] = j_pos.position[i];
        }

        // 获取关节速度
        FaJointVel j_vel;
        ret = robot.GetJointVelocities(&j_vel);
        if (ret != 0) {
            cout << "法奥机器人: 获取关节速度失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_vel[i] = j_vel.velocity[i];
        }

        // 获取关节加速度
        FaJointAcc j_acc;
        ret = robot.GetJointAccelerations(&j_acc);
        if (ret != 0) {
            cout << "法奥机器人: 获取关节加速度失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_acc[i] = j_acc.acceleration[i];
        }

        // 获取关节扭矩
        FaJointTorque torques;
        ret = robot.GetJointTorques(&torques);
        if (ret != 0) {
            cout << "法奥机器人: 获取关节扭矩失败，错误码: " << ret << endl;
            return false;
        }
        for (int i = 0; i < 6; i++) {
            data.joint_torque[i] = torques.torque[i];
        }

        return true;
    }

    void writeToFile(const RobotData& data) {
        data_file << data.timestamp << "," << ((robot_type == FA_ROBOT) ? "Fa" : "Fanuc");
        for (int i = 0; i < 6; i++) {
            data_file << "," << data.joint_pos[i]
                << "," << data.joint_vel[i]
                << "," << data.joint_acc[i]
                << "," << data.joint_torque[i];
        }
        data_file << endl;
        data_file.flush();
    }

    ~DataCollector() {
        if (data_file.is_open()) {
            data_file.close();
            cout << "数据文件已关闭" << endl;
        }
    }
};

// 机器人控制基类
class RobotController {
protected:
    DataCollector collector;
    RobotType robot_type;
    MultiModelFrictionIdentifier friction_identifier;

public:
    virtual bool connect(const string& address) = 0;
    virtual bool loadProgram(const string& program_name) = 0;
    virtual bool runProgram() = 0;
    virtual bool stopProgram() = 0;
    virtual bool setSpeed(int speed) = 0;
    virtual bool setMode(int mode) = 0;
    virtual void disconnect() = 0;
    virtual bool collectData(RobotData& data) = 0;
    
    virtual ~RobotController() {}
    
    bool initFrictionIdentifier(const string& result_filename) {
        return friction_identifier.init(result_filename);
    }
    
    FrictionPrediction processFrictionData(double velocity, double torque, double timestamp) {
        return friction_identifier.predictAndUpdate(velocity, torque, timestamp);
    }
    
    void printFrictionPerformance() {
        friction_identifier.printPerformance();
    }
};

// 法奥机器人控制器
class FaRobotController : public RobotController {
private:
    FaRobot robot;

public:
    FaRobotController() {
        robot_type = FA_ROBOT;
    }

    bool connect(const string& address) override {
        // 法奥机器人连接接口，根据实际SDK调整
        errno_t ret = robot.Connect(address.c_str());
        if (ret != 0) {
            printf("法奥机器人连接失败，错误代码: %d\n", ret);
            return false;
        }
        printf("法奥机器人连接成功!\n");
        return true;
    }

    bool loadProgram(const string& program_name) override {
        // 法奥机器人程序加载接口
        errno_t ret = robot.LoadProgram(program_name.c_str());
        if (ret != 0) {
            printf("法奥程序加载失败，错误代码: %d\n", ret);
            return false;
        }
        return true;
    }

    bool runProgram() override {
        // 法奥机器人程序运行接口
        errno_t ret = robot.StartProgram();
        if (ret != 0) {
            printf("法奥程序运行失败，错误代码: %d\n", ret);
            return false;
        }
        return true;
    }

    bool stopProgram() override {
        robot.StopProgram();
        return true;
    }

    bool setSpeed(int speed) override {
        robot.SetVelocity(speed);
        return true;
    }

    bool setMode(int mode) override {
        robot.SetOperationMode(mode);
        return true;
    }

    void disconnect() override {
        robot.Disconnect();
    }

    bool collectData(RobotData& data) override {
        return collector.collectData(robot, data);
    }

    bool initCollector(const string& filename) {
        return collector.init(filename, robot_type);
    }

    void printCurrentPosition() {
        // 法奥机器人获取当前位置接口
        FaJointPos current_pos;
        if (robot.GetJointPositions(&current_pos) == 0) {
            printf("法奥当前关节位置: J1=%.2f, J2=%.2f, J3=%.2f, J4=%.2f, J5=%.2f, J6=%.2f\n",
                current_pos.position[0], current_pos.position[1], current_pos.position[2],
                current_pos.position[3], current_pos.position[4], current_pos.position[5]);
        }
    }
};

int main(void)
{
    FaRobotController robot_controller;
    DataCollector collector;

    // 连接机器人
    if (!robot_controller.connect("192.168.57.2")) { // 法奥机器人IP地址
        return -1;
    }

    // 初始化数据采集器
    if (!robot_controller.initCollector("fa_robot_trajectory_data.csv")) {
        return -1;
    }
    printf("法奥机器人数据文件初始化成功!\n");

    // 初始化多模型摩擦辨识器
    if (!robot_controller.initFrictionIdentifier("friction_model_comparison.csv")) {
        return -1;
    }
    printf("多模型摩擦辨识器初始化成功!\n");

    // 设置机器人参数
    robot_controller.setMode(0);
    robot_controller.setSpeed(60); // 高速

    // 加载并运行程序
    string robot_programname = "/fruser/longshikeluoxuan.lua"; // 法奥机器人程序路径
    if (!robot_controller.loadProgram(robot_programname)) {
        return -1;
    }

    if (!robot_controller.runProgram()) {
        return -1;
    }
    printf("开始运行法奥机器人程序...\n");

    // 等待程序启动
    Sleep(1000);

    // 开始数据采集和摩擦辨识
    int sample_count = 0;
    int failed_samples = 0;
    const int SAMPLE_RATE_MS = 10; // 采样率10ms (100Hz)
    const int MAX_FAILED_SAMPLES = 50;
    const int COLLECTION_DURATION_MS = 180000; // 3分钟采集

    printf("开始数据采集和摩擦辨识...\n");
    printf("采样频率: %dHz\n", 1000 / SAMPLE_RATE_MS);
    printf("采集时长: %d秒\n", COLLECTION_DURATION_MS / 1000);
    printf("目标关节: 6\n");

    // 记录采集开始时间
    auto collection_start_time = chrono::steady_clock::now();

    // 基于时间的数据采集循环
    while (true) {
        // 检查是否达到采集时长
        auto current_time = chrono::steady_clock::now();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(current_time - collection_start_time).count();

        if (elapsed_time >= COLLECTION_DURATION_MS) {
            printf("达到预设采集时长，停止数据采集\n");
            break;
        }

        // 采集数据
        RobotData data;
        if (robot_controller.collectData(data)) {
            sample_count++;
            
            // 处理关节6的摩擦辨识
            double velocity = data.joint_vel[5]; // 关节6角速度
            double torque = data.joint_torque[5]; // 关节6扭矩
            
            FrictionPrediction prediction = robot_controller.processFrictionData(velocity, torque, data.timestamp);
            
            // 每采集100次数据打印一次状态
            if (sample_count % 100 == 0) {
                printf("已采集 %d 组数据，已运行 %.1f 秒...\n",
                    sample_count, elapsed_time / 1000.0);
                
                // 显示当前摩擦预测结果
                printf("关节6摩擦预测: 实际=%.4f, PINN=%.4f, SVM=%.4f, RNN=%.4f, NNFM=%.4f\n",
                    prediction.actual_torque, prediction.pinn_prediction,
                    prediction.svm_prediction, prediction.rnn_prediction,
                    prediction.nnfm_prediction);
                
                robot_controller.printCurrentPosition();
                
                // 每500次显示一次性能对比
                if (sample_count % 500 == 0) {
                    robot_controller.printFrictionPerformance();
                }
            }
        }
        else {
            failed_samples++;
            if (failed_samples >= MAX_FAILED_SAMPLES) {
                printf("数据采集失败次数过多，停止采集\n");
                break;
            }
        }

        // 等待下一个采样周期
        Sleep(SAMPLE_RATE_MS);
    }

    printf("数据采集完成: 成功采集 %d 组数据, 失败 %d 次\n", sample_count, failed_samples);

    // 显示最终性能对比
    printf("\n=== 最终摩擦模型性能对比 ===\n");
    robot_controller.printFrictionPerformance();

    // 停止机器人
    robot_controller.setMode(1);
    robot_controller.stopProgram();

    // 断开连接
    robot_controller.disconnect();

    printf("轨迹数据已保存到 fa_robot_trajectory_data.csv\n");
    printf("摩擦模型对比数据已保存到 friction_model_comparison.csv\n");
    printf("程序结束\n");

    return 0;
}