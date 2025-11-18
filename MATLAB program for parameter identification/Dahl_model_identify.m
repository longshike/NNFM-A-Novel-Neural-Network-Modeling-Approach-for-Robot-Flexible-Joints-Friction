% 7关节Dahl摩擦模型参数辨识脚本（基于revised_manuscript.pdf）
% 模型依据：稿件1-38至1-40段Dahl模型：F_fi^Dahl=σ₀z；ż=q̇·sgn(γ)·|γ|^β；γ=1-sgn(q̇)·σ₀z/Fc
% 数据处理：贴合稿件实验逻辑，保留必要数据清洗，确保参数辨识基于高质量数据
clear; clc; close all;

%% 1. 读取机器人7关节原始数据（适配稿件1-127至1-128段实验数据要求）
fprintf('读取机器人7关节实验数据...\n');
data = readtable('test_data_carsimotion_kuaisu.csv');  % 原始数据文件，需与实际路径一致

% 显示数据基本信息（验证数据加载有效性，符合稿件实验可复现性要求）
fprintf('数据列数: %d, 总数据点数: %d\n', width(data), height(data));
fprintf('模型依据：revised_manuscript.pdf 第1-38至1-40段Dahl摩擦模型\n');

%% 2. 初始化参数存储结构与核心配置（适配7关节需求）
joint_names = {'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'};  % 关节标识，需与CSV列名匹配
dahl_params = struct();  % 存储各关节Dahl模型参数：σ₀(刷毛刚度)、Fc(Coulomb摩擦)、β(滞回系数)
dt = 1/1000;  % 采样时间(s)，基于稿件1-128段常见机器人采样频率1000Hz设定

%% 3. 逐关节处理数据并进行Dahl模型参数辨识（贴合稿件动态模型辨识逻辑）
for i = 1:7
    fprintf('\n=== 开始处理关节 %d ===\n', i);
    
    % 3.1 提取当前关节的原始数据（角速度q̇、扭矩τ，列名需与CSV严格匹配）
    % 假设CSV列名格式：角速度'dq1'-'dq7'，扭矩'q1_tau_J_compensate'-'q7_tau_J_compensate'
    dq_raw = data.(['dq' num2str(i)]);  % 原始关节角速度 (rad/s)，对应稿件中q̇
    tau_raw = data.(['q' num2str(i) '_tau_J_compensate']);  % 原始关节扭矩 (Nm)，对应稿件中F_fi^Dahl
    
    % 3.2 数据清洗（仅移除NaN与极端异常值，不改变原始摩擦动态特性，符合稿件1-128段）
    % 异常值阈值：参考机器人关节常规范围，避免数据采集错误影响辨识
    valid_idx = ~isnan(dq_raw) & ~isnan(tau_raw) & ...
                abs(dq_raw) < 10 & abs(tau_raw) < 50;
    dq_valid = dq_raw(valid_idx);  % 清洗后的角速度序列
    tau_valid = tau_raw(valid_idx);  % 清洗后的扭矩序列
    n_valid = sum(valid_idx);  % 有效数据点数
    
    % 验证有效数据量（确保动态模型辨识可靠性，避免数据不足导致拟合失败）
    fprintf('关节%d数据清洗结果：有效数据点数 %d / 总数据点数 %d\n', ...
        i, n_valid, length(dq_raw));
    if n_valid < 100  % 动态模型需足够数据捕捉滞回特性，阈值高于静态模型
        fprintf('警告：关节 %d 有效数据点过少（<100个），跳过该关节辨识\n', i);
        continue;
    end
    
    % 3.3 Dahl模型参数辨识（基于动态模型拟合，严格遵循稿件1-39段公式）
    % 步骤1：定义Dahl动态模型（输入参数[σ₀,Fc,β]，输出扭矩预测值）
    dahl_dynamic_model = @(params, dq_series, dt) ...
        dahl_friction_model_dynamic(dq_series, dt, params(1), params(2), params(3));
    
    % 步骤2：设置参数初始猜测与边界（基于稿件1-40段参数物理意义设定）
    % σ₀(刷毛刚度)：1-1000 Nm/rad；Fc(Coulomb摩擦)：0.01-10 Nm；β(滞回系数)：0.1-2（稿件1-40段）
    initial_guess = [50, 0.5, 0.8];  % 初始猜测，可根据机器人类型微调
    lb = [1, 0.01, 0.1];   % 参数下界
    ub = [1000, 10, 2];     % 参数上界
    
    % 步骤3：非线性最小二乘拟合（最小化预测扭矩与实际扭矩误差，贴合稿件1-143段模型优化逻辑）
    options = optimoptions('lsqcurvefit', ...
        'Display', 'off', ...
        'MaxFunctionEvaluations', 10000, ...
        'MaxIterations', 2000, ...
        'TolFun', 1e-6);  % 提升拟合精度，符合稿件高精度建模需求
    
    try
        % 调用拟合函数，基于动态模型输出的扭矩序列进行参数优化
        [params_opt, resnorm] = lsqcurvefit(...
            @(p, x) dahl_dynamic_model(p, x, dt), ...
            initial_guess, ...
            dq_valid, ...
            tau_valid, ...
            lb, ub, options);
        
        % 提取辨识后的Dahl模型参数（对应稿件1-40段定义）
        sigma0 = params_opt(1);  % σ₀：刷毛平均刚度系数 (Nm/rad)
        Fc = params_opt(2);      % Fc：Coulomb摩擦系数 (Nm)
        beta = params_opt(3);    % β：滞回环系数（决定滞回曲线形状）
        
        % 3.4 模型性能评估（使用稿件1-131段明确的性能指标PI：RMS、R²）
        tau_pred = dahl_dynamic_model(params_opt, dq_valid, dt);  % 模型预测扭矩
        rmse = sqrt(mean((tau_valid - tau_pred).^2));  % 均方根误差
        ss_res = sum((tau_valid - tau_pred).^2);       % 残差平方和
        ss_tot = sum((tau_valid - mean(tau_valid)).^2);% 总平方和
        R_squared = 1 - (ss_res / ss_tot);             % 决定系数（拟合优度）
        
        fprintf('关节%d Dahl模型参数辨识成功！\n', i);
        
    catch ME
        % 拟合失败时的备选逻辑（避免脚本中断，提供基础参数参考）
        fprintf('关节%d Dahl模型拟合失败：%s\n', i, ME.message);
        fprintf('启用备选参数（基于物理意义设定）...\n');
        sigma0 = 50;    % 默认刷毛刚度
        Fc = 0.5;       % 默认Coulomb摩擦
        beta = 1.0;     % 默认滞回系数
        tau_pred = dahl_dynamic_model([sigma0, Fc, beta], dq_valid, dt);
        rmse = sqrt(mean((tau_valid - tau_pred).^2));
        R_squared = 1 - (sum((tau_valid - tau_pred).^2) / ss_tot);
    end
    
    % 3.5 存储当前关节的Dahl模型参数与性能指标（便于后续调用与验证）
    dahl_params(i).joint_id = i;          % 关节编号
    dahl_params(i).sigma0 = sigma0;       % 刷毛刚度系数 (Nm/rad)，稿件1-40段σ₀
    dahl_params(i).Fc = Fc;               % Coulomb摩擦系数 (Nm)，稿件1-40段Fc
    dahl_params(i).beta = beta;           % 滞回环系数，稿件1-40段β
    dahl_params(i).R_squared = R_squared; % 决定系数（拟合优度）
    dahl_params(i).rmse = rmse;           % RMS误差 (Nm)
    dahl_params(i).num_samples = n_valid; % 有效数据点数
    dahl_params(i).dt = dt;               % 采样时间 (s)，用于后续动态模型调用
    
    % 输出辨识结果（格式贴合稿件中模型参数展示）
    fprintf('关节 %d Dahl模型参数：\n', i);
    fprintf('  刷毛刚度系数 σ₀ = %.6f Nm/rad\n', sigma0);
    fprintf('  Coulomb摩擦系数 Fc = %.6f Nm\n', Fc);
    fprintf('  滞回环系数 β = %.6f\n', beta);
    fprintf('  模型性能：R² = %.4f, RMS误差 = %.6f Nm\n', R_squared, rmse);
end

%% 4. 绘制各关节扭矩对比图（动态模型预测vs原始测量，贴合稿件1-143段实验可视化）
fprintf('\n绘制7关节Dahl模型扭矩对比图...\n');
figure('Position', [100, 100, 1400, 900]);  % 设置画布大小，适配7关节子图

for i = 1:7
    % 跳过未成功辨识参数的关节
    if ~isfield(dahl_params(i), 'joint_id') || isempty(dahl_params(i).joint_id)
        subplot(3, 3, i);
        axis off;
        text(0.5, 0.5, sprintf('关节 %d\n无有效辨识结果', i), 'HorizontalAlignment', 'center', 'FontSize', 10);
        continue;
    end
    
    % 4.1 提取当前关节的清洗后数据（与辨识时一致）
    dq_raw = data.(['dq' num2str(i)]);
    tau_raw = data.(['q' num2str(i) '_tau_J_compensate']);
    valid_idx = ~isnan(dq_raw) & ~isnan(tau_raw) & ...
                abs(dq_raw) < 10 & abs(tau_raw) < 50;
    dq_plot = dq_raw(valid_idx);
    tau_plot = tau_raw(valid_idx);
    
    % 4.2 基于辨识参数计算Dahl模型预测扭矩（动态版本，贴合稿件1-39段公式）
    sigma0 = dahl_params(i).sigma0;
    Fc = dahl_params(i).Fc;
    beta = dahl_params(i).beta;
    tau_pred = dahl_friction_model_dynamic(dq_plot, dt, sigma0, Fc, beta);
    
    % 4.3 绘制对比图（原始测量扭矩+动态模型预测扭矩，标注核心参数）
    subplot(3, 3, i);
    % 原始测量扭矩（蓝色散点，体现原始摩擦动态）
    plot(1:length(tau_plot), tau_plot, 'b-', 'LineWidth', 1, 'DisplayName', '原始测量扭矩');
    hold on;
    % Dahl模型预测扭矩（红色实线，体现动态拟合效果）
    plot(1:length(tau_pred), tau_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Dahl模型预测扭矩');
    
    % 图表标注（符合稿件实验可视化规范，明确参数与性能）
    xlabel('数据点序号', 'FontSize', 9);
    ylabel('关节扭矩 (Nm)', 'FontSize', 9);
    title(sprintf('关节 %d\nσ₀=%.1f, Fc=%.3f, R²=%.3f', ...
        i, sigma0, Fc, dahl_params(i).R_squared), 'FontSize', 10);
    grid on;
    legend('Location', 'best', 'FontSize', 8);
    hold off;
end

% 4.4 添加参数总结子图（便于快速查看所有关节核心参数，贴合稿件实验报告逻辑）
subplot(3, 3, 8);
axis off;
text(0.1, 0.95, '7关节Dahl模型参数总结', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
y_pos = 0.85;  % 文本垂直位置初始化
for i = 1:7
    if isfield(dahl_params(i), 'joint_id') && ~isempty(dahl_params(i).joint_id)
        text(0.1, y_pos, sprintf('关节 %d: σ₀=%.1f, Fc=%.3f, β=%.2f', ...
            i, dahl_params(i).sigma0, dahl_params(i).Fc, dahl_params(i).beta), 'FontSize', 10, 'HorizontalAlignment', 'left');
        y_pos = y_pos - 0.1;
        if y_pos < 0.1  % 避免文本超出子图范围
            break;
        end
    end
end

% 总标题（明确标注模型来源，符合脚本核心逻辑）
sgtitle('7关节Dahl摩擦模型扭矩对比（基于revised_manuscript.pdf第1-38至1-40段）', 'FontSize', 14, 'FontWeight', 'bold');

%% 5. 保存Dahl模型参数到MAT文件（便于后续调用验证，符合稿件1-127段实验可复现性要求）
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
mat_filename = sprintf('dahl_params_7joints_%s.mat', timestamp);

% 保存参数结构（包含所有关节的参数、性能指标与采样时间）
save(mat_filename, 'dahl_params');
fprintf('\n7关节Dahl模型参数已保存至：%s\n', fullfile(pwd(), mat_filename));
fprintf('保存内容：σ₀(刷毛刚度)、Fc(Coulomb摩擦)、β(滞回系数)、R²、RMS误差\n');

%% 6. 核心函数定义（严格遵循稿件1-38至1-40段Dahl模型公式）
function [tau_friction, z_history] = dahl_friction_model_dynamic(dq_series, dt, sigma0, Fc, beta)
    % Dahl摩擦模型动态实现（完全匹配revised_manuscript.pdf第1-39至1-40段公式）
    % 公式参考：
    % 1. F_fi^Dahl = σ₀·z （摩擦扭矩与刷毛平均挠度成正比）
    % 2. ż = q̇·sgn(γ)·|γ|^β （刷毛挠度变化率，体现滞回特性）
    % 3. γ = 1 - sgn(q̇)·σ₀·z/Fc （中间变量，关联挠度与Coulomb摩擦）
    %
    % 输入：
    %   dq_series - 角速度时间序列 (rad/s)，对应稿件中q̇
    %   dt - 采样时间 (s)
    %   sigma0 - 刷毛刚度系数 (Nm/rad)，对应稿件中σ₀
    %   Fc - Coulomb摩擦系数 (Nm)，对应稿件中Fc
    %   beta - 滞回环系数，对应稿件中β
    % 输出：
    %   tau_friction - 摩擦扭矩预测序列 (Nm)，对应稿件中F_fi^Dahl
    %   z_history - 刷毛平均挠度历史序列 (rad)，对应稿件中z
    
    n = length(dq_series);
    z_history = zeros(n, 1);  % 初始化刷毛挠度序列
    tau_friction = zeros(n, 1);  % 初始化摩擦扭矩序列
    z = 0;  % 初始挠度（静止状态下挠度为0，符合稿件物理假设）
    
    for k = 1:n
        dq = dq_series(k);  % 当前时刻角速度
        
        % 计算中间变量γ（稿件1-39段公式3）
        if abs(dq) > 1e-6  % 角速度非零，避免除以零
            gamma = 1 - sign(dq) * sigma0 * z / Fc;
        else
            gamma = 1;  % 角速度为零时，γ=1，挠度变化率为0
        end
        
        % 计算刷毛挠度变化率ż（稿件1-39段公式2）
        if abs(gamma) < 1e-6
            dz_dt = 0;
        else
            dz_dt = dq * sign(gamma) * abs(gamma)^beta;
        end
        
        % 更新刷毛挠度z（欧拉积分，基于采样时间dt）
        z = z + dz_dt * dt;
        
        % 物理约束：挠度z的绝对值不超过Fc/sigma0（避免摩擦扭矩超过Coulomb摩擦，符合稿件1-40段）
        z = max(min(z, Fc/sigma0), -Fc/sigma0);
        
        % 计算摩擦扭矩（稿件1-39段公式1）
        tau_friction(k) = sigma0 * z;
        z_history(k) = z;
    end
end

function tau_friction = dahl_friction_model_static(dq, params)
    % Dahl摩擦模型静态近似（适配无时间序列场景，基于稿件1-40段特性推导）
    % 输入：
    %   dq - 角速度 (rad/s)
    %   params - 模型参数[sigma0, Fc, beta]
    % 输出：
    %   tau_friction - 静态摩擦扭矩预测值 (Nm)
    
    sigma0 = params(1);
    Fc = params(2);
    beta = params(3);
    
    if abs(dq) < 1e-6
        tau_friction = 0;
    else
        % 基于动态模型稳态特性推导的静态近似公式，贴合稿件中摩擦与角速度关系
        z_static = (Fc/sigma0) * (1 - exp(-abs(dq)/beta));  % 稳态挠度近似
        tau_friction = sigma0 * z_static * sign(dq);
    end
end

