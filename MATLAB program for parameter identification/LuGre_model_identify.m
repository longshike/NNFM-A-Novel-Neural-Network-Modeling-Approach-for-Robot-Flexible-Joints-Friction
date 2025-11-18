%% 7关节LuGre摩擦模型参数辨识脚本（使用原始数据）
clear; clc; close all;

%% 1. 读取数据
fprintf('读取机器人7关节数据...\n');
data = readtable('test_data_carsimotion_kuaisu.csv');  % 修改为您的文件名

% 显示数据信息
fprintf('数据列数: %d, 数据点数: %d\n', width(data), height(data));

%% 2. 处理各关节数据（直接使用原始数据，不进行滤波）
joint_names = {'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'};
lugre_params = struct();

for i = 1:7
    fprintf('\n=== 处理关节 %d ===\n', i);
    
    % 提取原始数据（根据实际数据列名调整）
    % 假设数据列名格式：dq1, dq2, ..., dq7 为角速度
    % q1_tau_J_compensate, q2_tau_J_compensate, ... 为扭矩
    dq_raw = data.(['dq' num2str(i)]);
    tau_raw = data.(['q' num2str(i) '_tau_J_compensate']);
    
    % 直接使用原始数据，不进行滤波
    dq_used = dq_raw;
    tau_used = tau_raw;
    
    % 数据预处理（只去除异常值和NaN）
    valid_idx = ~isnan(dq_used) & ~isnan(tau_used) & ...
                abs(dq_used) < 10 & abs(tau_used) < 50;
    dq_valid = dq_used(valid_idx);
    tau_valid = tau_used(valid_idx);
    
    fprintf('原始数据点数: %d, 有效数据点数: %d\n', length(dq_raw), sum(valid_idx));
    
    if sum(valid_idx) < 10
        fprintf('警告: 关节 %d 有效数据点过少，跳过\n', i);
        continue;
    end
    
    %% 3. LuGre模型参数辨识
    % LuGre模型: 
    % F = σ0*z + σ1*dz/dt + σ2*dq
    % dz/dt = dq - |dq|/g(dq) * z
    % g(dq) = [Fc + (Fs - Fc)*exp(-(dq/vs)^2)] / σ0
    
    % 设置初始参数猜测
    % [σ0, σ1, σ2, Fc, Fs, vs]
    initial_guess = [100, 1, 0.05, 0.1, 0.15, 0.1];  % 根据实际情况调整
    
    % 定义LuGre模型函数（稳态近似）
    lugre_model_static = @(params, dq) steady_state_lugre(params, dq);
    
    % 设置参数边界
    lb = [1, 0.01, 0, 0.01, 0.01, 0.01];   % 下界
    ub = [1000, 100, 1, 10, 10, 1];         % 上界
    
    % 非线性最小二乘拟合
    options = optimoptions('lsqcurvefit', 'Display', 'iter', ...
        'MaxFunctionEvaluations', 5000, 'MaxIterations', 1000);
    
    try
        params_opt = lsqcurvefit(lugre_model_static, initial_guess, ...
            dq_valid, tau_valid, lb, ub, options);
        
        sigma0 = params_opt(1);
        sigma1 = params_opt(2);
        sigma2 = params_opt(3);
        Fc = params_opt(4);
        Fs = params_opt(5);
        vs = params_opt(6);
        
        % 计算预测值和性能指标
        tau_pred = lugre_model_static(params_opt, dq_valid);
        rmse = sqrt(mean((tau_valid - tau_pred).^2));
        ss_res = sum((tau_valid - tau_pred).^2);
        ss_tot = sum((tau_valid - mean(tau_valid)).^2);
        R_squared = 1 - (ss_res / ss_tot);
        
        fprintf('LuGre模型拟合成功!\n');
        
    catch ME
        fprintf('LuGre模型拟合失败: %s\n', ME.message);
        % 使用Stribeck模型作为备选
        fprintf('使用Stribeck模型作为备选...\n');
        
        % 先拟合Stribeck模型
        stribeck_model = @(params, dq) params(3)*dq + ...
            sign(dq).*(params(1) + (params(2)-params(1)).*exp(-(abs(dq)/params(4)).^params(5)));
        
        stribeck_guess = [0.1, 0.15, 0.05, 0.1, 1.0];
        stribeck_lb = [0, 0, 0, 0.01, 0.5];
        stribeck_ub = [10, 10, 1, 1, 2];
        
        try
            stribeck_params = lsqcurvefit(stribeck_model, stribeck_guess, ...
                dq_valid, tau_valid, stribeck_lb, stribeck_ub, options);
            
            Fc = stribeck_params(1);
            Fs = stribeck_params(2);
            sigma2 = stribeck_params(3);
            vs = stribeck_params(4);
            % 设置默认的σ0和σ1
            sigma0 = 100;
            sigma1 = 1.0;
            
            tau_pred = lugre_model_static([sigma0, sigma1, sigma2, Fc, Fs, vs], dq_valid);
            rmse = sqrt(mean((tau_valid - tau_pred).^2));
            R_squared = 1 - (sum((tau_valid - tau_pred).^2) / sum((tau_valid - mean(tau_valid)).^2));
            
        catch
            % 最后回退到CV模型
            fprintf('使用CV模型作为最后备选...\n');
            X = [sign(dq_valid), dq_valid];
            theta = (X' * X) \ (X' * tau_valid);
            Fc = theta(1);
            Fs = Fc * 1.5;
            sigma2 = theta(2);
            sigma0 = 100;
            sigma1 = 1.0;
            vs = 0.1;
            
            tau_pred = lugre_model_static([sigma0, sigma1, sigma2, Fc, Fs, vs], dq_valid);
            rmse = sqrt(mean((tau_valid - tau_pred).^2));
            R_squared = 1 - (sum((tau_valid - tau_pred).^2) / sum((tau_valid - mean(tau_valid)).^2));
        end
    end
    
    %% 4. 保存参数
    lugre_params(i).joint_id = i;
    lugre_params(i).sigma0 = sigma0;
    lugre_params(i).sigma1 = sigma1;
    lugre_params(i).sigma2 = sigma2;
    lugre_params(i).Fc = Fc;
    lugre_params(i).Fs = Fs;
    lugre_params(i).vs = vs;
    lugre_params(i).R_squared = R_squared;
    lugre_params(i).rmse = rmse;
    lugre_params(i).num_samples = sum(valid_idx);
    
    fprintf('关节 %d LuGre参数:\n', i);
    fprintf('  σ0 = %.6f, σ1 = %.6f, σ2 = %.6f\n', sigma0, sigma1, sigma2);
    fprintf('  Fc = %.6f, Fs = %.6f, vs = %.6f\n', Fc, Fs, vs);
    fprintf('  R² = %.4f, RMSE = %.6f\n', R_squared, rmse);
end

%% 5. 绘制每个关节的扭矩对比图
figure('Position', [100, 100, 1400, 900]);

for i = 1:7
    if isempty(lugre_params(i).joint_id)
        continue;
    end
    
    subplot(3, 3, i);
    
    % 提取数据（直接使用原始数据）
    dq_raw = data.(['dq' num2str(i)]);
    tau_raw = data.(['q' num2str(i) '_tau_J_compensate']);
    
    % 准备绘图数据
    valid_idx = ~isnan(dq_raw) & ~isnan(tau_raw) & ...
                abs(dq_raw) < 10 & abs(tau_raw) < 50;
    dq_plot = dq_raw(valid_idx);
    tau_plot = tau_raw(valid_idx);
    
    % 按角速度排序
    [dq_sorted, sort_idx] = sort(dq_plot);
    tau_sorted = tau_plot(sort_idx);
    
    % 计算模型预测
    params = [lugre_params(i).sigma0, lugre_params(i).sigma1, lugre_params(i).sigma2, ...
              lugre_params(i).Fc, lugre_params(i).Fs, lugre_params(i).vs];
    tau_pred = steady_state_lugre(params, dq_sorted);
    
    % 绘制对比图
    plot(dq_sorted, tau_sorted, 'b.', 'MarkerSize', 8, 'DisplayName', '测量扭矩');
    hold on;
    plot(dq_sorted, tau_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'LuGre模型预测');
    
    xlabel('角速度 (rad/s)');
    ylabel('扭矩 (Nm)');
    title(sprintf('关节 %d: σ0=%.1f, σ1=%.1f, R²=%.3f', i, ...
        lugre_params(i).sigma0, lugre_params(i).sigma1, lugre_params(i).R_squared));
    grid on;
    legend('Location', 'best');
end

% 添加总结信息
subplot(3, 3, 8);
axis off;
text(0.1, 0.9, 'LuGre摩擦模型参数总结:', 'FontSize', 12, 'FontWeight', 'bold');
y_pos = 0.8;
for i = 1:7
    if ~isempty(lugre_params(i).joint_id)
        text(0.1, y_pos, sprintf('关节 %d: σ0=%.1f, σ1=%.1f', i, ...
            lugre_params(i).sigma0, lugre_params(i).sigma1), 'FontSize', 10);
        y_pos = y_pos - 0.1;
    end
end

sgtitle('7关节LuGre摩擦模型扭矩对比（使用原始数据）');

%% 6. 保存参数到文件
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
mat_filename = sprintf('lugre_params_7joints_%s.mat', timestamp);

% 保存MAT文件
save(mat_filename, 'lugre_params');
fprintf('\n参数已保存至: %s\n', mat_filename);

%% 7. 使用说明
fprintf('\n=== 使用说明 ===\n');
fprintf('下次使用时，加载MAT文件并调用lugre_friction_model函数:\n');
fprintf('load(''%s'');\n', mat_filename);
fprintf('dq_test = 0.5; %% 测试角速度\n');
fprintf('joint_id = 1; %% 关节编号\n');
fprintf('tau_friction = lugre_friction_model(dq_test, lugre_params, joint_id);\n');

%% 函数定义部分
function tau_friction = lugre_friction_model(dq, params, joint_id)
    % LuGre摩擦模型（稳态近似）
    % F = σ0*z + σ1*dz/dt + σ2*dq
    % 在稳态下，dz/dt=0，z = g(dq)*sign(dq)
    % g(dq) = [Fc + (Fs - Fc)*exp(-(dq/vs)^2)] / σ0
    %
    % 输入:
    %   dq - 角速度 (rad/s)
    %   params - 参数结构体数组
    %   joint_id - 关节编号 (1-7)
    % 输出:
    %   tau_friction - 摩擦扭矩预测值 (Nm)
    
    if joint_id < 1 || joint_id > 7 || isempty(params(joint_id).joint_id)
        error('无效的关节编号或该关节参数未辨识');
    end
    
    sigma0 = params(joint_id).sigma0;
    sigma1 = params(joint_id).sigma1;
    sigma2 = params(joint_id).sigma2;
    Fc = params(joint_id).Fc;
    Fs = params(joint_id).Fs;
    vs = params(joint_id).vs;
    
    % 计算LuGre摩擦（稳态近似）
    g_dq = (Fc + (Fs - Fc) * exp(-(dq/vs).^2)) / sigma0;
    z_steady = g_dq .* sign(dq);
    dz_dt_steady = zeros(size(dq));  % 稳态下dz/dt=0
    
    tau_friction = sigma0 * z_steady + sigma1 * dz_dt_steady + sigma2 * dq;
end

function tau_steady = steady_state_lugre(params, dq)
    % LuGre模型稳态近似计算
    % 用于参数辨识
    sigma0 = params(1);
    sigma1 = params(2);
    sigma2 = params(3);
    Fc = params(4);
    Fs = params(5);
    vs = params(6);
    
    % 计算稳态摩擦
    g_dq = (Fc + (Fs - Fc) * exp(-(abs(dq)/vs).^2)) / sigma0;
    z_steady = g_dq .* sign(dq);
    dz_dt_steady = zeros(size(dq));  % 稳态下dz/dt=0
    
    tau_steady = sigma0 * z_steady + sigma1 * dz_dt_steady + sigma2 * dq;
end

% 完整的LuGre模型（动态版本，需要时间序列数据）
function [tau_friction, z_history] = lugre_friction_model_dynamic(dq_series, dt, params, joint_id)
    % LuGre摩擦模型（动态版本）
    % F = σ0*z + σ1*dz/dt + σ2*dq
    % dz/dt = dq - |dq|/g(dq) * z
    % g(dq) = [Fc + (Fs - Fc)*exp(-(dq/vs)^2)] / σ0
    %
    % 输入:
    %   dq_series - 角速度时间序列 (rad/s)
    %   dt - 采样时间 (s)
    %   params - 参数结构体数组
    %   joint_id - 关节编号 (1-7)
    % 输出:
    %   tau_friction - 摩擦扭矩预测值序列 (Nm)
    %   z_history - 内部状态z的历史值
    
    if joint_id < 1 || joint_id > 7 || isempty(params(joint_id).joint_id)
        error('无效的关节编号或该关节参数未辨识');
    end
    
    sigma0 = params(joint_id).sigma0;
    sigma1 = params(joint_id).sigma1;
    sigma2 = params(joint_id).sigma2;
    Fc = params(joint_id).Fc;
    Fs = params(joint_id).Fs;
    vs = params(joint_id).vs;
    
    n = length(dq_series);
    z_history = zeros(n, 1);
    tau_friction = zeros(n, 1);
    
    % 初始状态
    z = 0;
    
    for k = 1:n
        dq = dq_series(k);
        
        % 计算g(dq)
        g_dq = (Fc + (Fs - Fc) * exp(-(abs(dq)/vs).^2)) / sigma0;
        
        % 更新内部状态z
        if abs(dq) > 1e-6  % 避免除以零
            dz_dt = dq - (abs(dq) / g_dq) * z;
        else
            dz_dt = - (1 / g_dq) * z;  % 当dq=0时的特殊处理
        end
        
        z = z + dz_dt * dt;
        
        % 计算摩擦扭矩
        tau_friction(k) = sigma0 * z + sigma1 * dz_dt + sigma2 * dq;
        z_history(k) = z;
    end
end

%% 8. 数据列名适配（根据实际CSV文件调整）
% 如果您的数据列名不同，请修改以下部分：
% 例如：
% - 如果角速度列名为 'velocity1', 'velocity2', ... 
%   修改: dq_raw = data.(['velocity' num2str(i)]);
%
% - 如果扭矩列名为 'torque1', 'torque2', ...
%   修改: tau_raw = data.(['torque' num2str(i)]);
%
% 请根据您的实际CSV文件列名进行调整

%%%%后续使用%%%
% 加载参数
% load('lugre_params_7joints_2024-01-01_12-00-00.mat');
% 
% % 使用LuGre模型（稳态近似）
% dq_test = 0.5;
% joint_id = 1;
% tau_friction = lugre_friction_model(dq_test, lugre_params, joint_id);
% 
% % 使用LuGre模型（动态版本，需要时间序列）
% dq_series = [0.1, 0.2, 0.3, 0.4, 0.5]; % 角速度时间序列
% dt = 0.001; % 采样时间
% [tau_friction_series, z_history] = lugre_friction_model_dynamic(dq_series, dt, lugre_params, joint_id);