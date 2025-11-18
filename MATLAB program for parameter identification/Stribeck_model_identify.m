%% 7关节Stribeck摩擦模型参数辨识脚本（使用原始数据）
clear; clc; close all;

%% 1. 读取数据
fprintf('读取机器人7关节数据...\n');
data = readtable('test_data_carsimotion_kuaisu.csv');  % 修改为您的文件名

% 显示数据信息
fprintf('数据列数: %d, 数据点数: %d\n', width(data), height(data));

%% 2. 处理各关节数据（直接使用原始数据，不进行滤波）
joint_names = {'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'};
stribeck_params = struct();

for i = 1:7
    fprintf('\n=== 处理关节 %d ===\n', i);
    
    % 提取原始数据（根据实际数据列名调整）
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
    
    %% 3. Stribeck模型参数辨识
    % Stribeck模型: tau = fv*dq + sign(dq)*[Fc + (Fs - Fc)*exp(-(dq/vs)^α)]
    
    % 设置初始参数猜测
    % [Fc, Fs, fv, vs, alpha]
    initial_guess = [0.1, 0.15, 0.05, 0.1, 1.0];  % 根据实际情况调整
    
    % 定义Stribeck模型函数
    stribeck_model = @(params, dq) params(3)*dq + ...
        sign(dq).*(params(1) + (params(2)-params(1)).*exp(-(abs(dq)/params(4)).^params(5)));
    
    % 设置参数边界
    lb = [0, 0, 0, 0.01, 0.5];   % 下界
    ub = [10, 10, 1, 1, 2];      % 上界
    
    % 非线性最小二乘拟合
    options = optimoptions('lsqcurvefit', 'Display', 'off', ...
        'MaxFunctionEvaluations', 5000, 'MaxIterations', 1000);
    
    try
        params_opt = lsqcurvefit(stribeck_model, initial_guess, ...
            dq_valid, tau_valid, lb, ub, options);
        
        Fc = params_opt(1);
        Fs = params_opt(2);
        fv = params_opt(3);
        vs = params_opt(4);
        alpha = params_opt(5);
        
        % 计算预测值和性能指标
        tau_pred = stribeck_model(params_opt, dq_valid);
        rmse = sqrt(mean((tau_valid - tau_pred).^2));
        ss_res = sum((tau_valid - tau_pred).^2);
        ss_tot = sum((tau_valid - mean(tau_valid)).^2);
        R_squared = 1 - (ss_res / ss_tot);
        
        fprintf('Stribeck模型拟合成功!\n');
        
    catch ME
        fprintf('Stribeck模型拟合失败: %s\n', ME.message);
        % 使用CV模型作为备选
        fprintf('使用CV模型作为备选...\n');
        X = [sign(dq_valid), dq_valid];
        theta = (X' * X) \ (X' * tau_valid);
        Fc = theta(1);
        Fs = Fc * 1.5;  % 假设静摩擦比动摩擦大50%
        fv = theta(2);
        vs = 0.1;
        alpha = 1.0;
        
        tau_pred = stribeck_model([Fc, Fs, fv, vs, alpha], dq_valid);
        rmse = sqrt(mean((tau_valid - tau_pred).^2));
        R_squared = 1 - (sum((tau_valid - tau_pred).^2) / sum((tau_valid - mean(tau_valid)).^2));
    end
    
    %% 4. 保存参数
    stribeck_params(i).joint_id = i;
    stribeck_params(i).Fc = Fc;
    stribeck_params(i).Fs = Fs;
    stribeck_params(i).fv = fv;
    stribeck_params(i).vs = vs;
    stribeck_params(i).alpha = alpha;
    stribeck_params(i).R_squared = R_squared;
    stribeck_params(i).rmse = rmse;
    stribeck_params(i).num_samples = sum(valid_idx);
    
    fprintf('关节 %d Stribeck参数:\n', i);
    fprintf('  Fc = %.6f, Fs = %.6f, fv = %.6f\n', Fc, Fs, fv);
    fprintf('  vs = %.6f, α = %.6f, R² = %.4f\n', vs, alpha, R_squared);
end

%% 5. 绘制每个关节的扭矩对比图
figure('Position', [100, 100, 1400, 900]);

for i = 1:7
    if isempty(stribeck_params(i).joint_id)
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
    params = [stribeck_params(i).Fc, stribeck_params(i).Fs, ...
              stribeck_params(i).fv, stribeck_params(i).vs, ...
              stribeck_params(i).alpha];
    tau_pred = stribeck_model(params, dq_sorted);
    
    % 绘制对比图
    plot(dq_sorted, tau_sorted, 'b.', 'MarkerSize', 8, 'DisplayName', '测量扭矩');
    hold on;
    plot(dq_sorted, tau_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'Stribeck模型预测');
    
    xlabel('角速度 (rad/s)');
    ylabel('扭矩 (Nm)');
    title(sprintf('关节 %d: Fc=%.3f, Fs=%.3f, R²=%.3f', i, ...
        stribeck_params(i).Fc, stribeck_params(i).Fs, stribeck_params(i).R_squared));
    grid on;
    legend('Location', 'best');
end

% 添加总结信息
subplot(3, 3, 8);
axis off;
text(0.1, 0.9, 'Stribeck摩擦模型参数总结:', 'FontSize', 12, 'FontWeight', 'bold');
y_pos = 0.8;
for i = 1:7
    if ~isempty(stribeck_params(i).joint_id)
        text(0.1, y_pos, sprintf('关节 %d: Fc=%.3f, Fs=%.3f', i, ...
            stribeck_params(i).Fc, stribeck_params(i).Fs), 'FontSize', 10);
        y_pos = y_pos - 0.1;
    end
end

sgtitle('7关节Stribeck摩擦模型扭矩对比（使用原始数据）');

%% 6. 保存参数到文件
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
mat_filename = sprintf('stribeck_params_7joints_%s.mat', timestamp);

% 保存MAT文件
save(mat_filename, 'stribeck_params');
fprintf('\n参数已保存至: %s\n', mat_filename);

%% 7. 使用说明
fprintf('\n=== 使用说明 ===\n');
fprintf('下次使用时，加载MAT文件并调用stribeck_friction_model函数:\n');
fprintf('load(''%s'');\n', mat_filename);
fprintf('dq_test = 0.5; %% 测试角速度\n');
fprintf('joint_id = 1; %% 关节编号\n');
fprintf('tau_friction = stribeck_friction_model(dq_test, stribeck_params, joint_id);\n');

%% 函数定义部分
function tau_friction = stribeck_friction_model(dq, params, joint_id)
    % Stribeck摩擦模型
    % tau = fv*dq + sign(dq)*[Fc + (Fs - Fc)*exp(-(dq/vs)^α)]
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
    
    Fc = params(joint_id).Fc;
    Fs = params(joint_id).Fs;
    fv = params(joint_id).fv;
    vs = params(joint_id).vs;
    alpha = params(joint_id).alpha;
    
    % 计算Stribeck摩擦
    stribeck_term = Fc + (Fs - Fc) * exp(-(abs(dq)/vs).^alpha);
    tau_friction = fv * dq + sign(dq) .* stribeck_term;
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

%%%%后续调用%%%%%
% 加载参数
% load('stribeck_params_7joints_2024-01-01_12-00-00.mat');
% 
% % 使用Stribeck模型
% dq_test = 0.5;
% joint_id = 1;
% tau_friction = stribeck_friction_model(dq_test, stribeck_params, joint_id);