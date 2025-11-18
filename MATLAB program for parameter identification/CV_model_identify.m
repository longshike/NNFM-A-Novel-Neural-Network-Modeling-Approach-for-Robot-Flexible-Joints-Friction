% 7关节CV摩擦模型参数辨识脚本（无滤波版，基于revised_manuscript.pdf）
% 模型依据：稿件1-36至1-37段Coulomb和粘性摩擦（CV）模型：F_fi^CV = fv*q_dot + Fc*sign(q_dot)
% 数据处理：仅使用原始数据，移除巴特沃斯低通滤波器，保留核心参数辨识与扭矩对比
clear; clc; close all;

%% 1. 读取原始机器人7关节数据（无滤波，直接使用原始数据）
fprintf('读取机器人7关节原始数据...\n');
data = readtable('test_data_carsimotion_kuaisu.csv');  % 原始数据文件路径，需与实际一致

% 显示原始数据基本信息（验证数据加载有效性）
fprintf('原始数据列数: %d, 原始数据点数: %d\n', width(data), height(data));
fprintf('注：本脚本移除巴特沃斯滤波，直接使用原始数据进行CV模型参数辨识\n');

%% 2. 初始化参数存储结构与关节名称（适配7关节需求）
joint_names = {'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7'};  % 关节标识，需与CSV列名匹配
cv_params = struct();  % 存储各关节CV模型参数（Fc：Coulomb摩擦系数，fv：粘性摩擦系数）

%% 3. 逐关节处理原始数据并进行CV模型参数辨识
for i = 1:7
    fprintf('\n=== 开始处理关节 %d（基于原始数据） ===\n', i);
    
    % 3.1 提取当前关节的原始角速度与原始扭矩数据（列名需与CSV文件严格匹配）
    % 假设CSV列名格式：角速度为'dq1'-'dq7'，扭矩为'q1_tau_J_compensate'-'q7_tau_J_compensate'
    dq_raw = data.(['dq' num2str(i)]);  % 原始关节角速度（rad/s）
    tau_raw = data.(['q' num2str(i) '_tau_J_compensate']);  % 原始关节扭矩（Nm）
    
    % 3.2 原始数据清洗（仅移除NaN值与极端异常值，不进行滤波）
    % 异常值阈值：角速度绝对值<10 rad/s（避免数据采集错误），扭矩绝对值<50 Nm（符合机器人关节扭矩范围）
    valid_idx = ~isnan(dq_raw) & ~isnan(tau_raw) & ...
                abs(dq_raw) < 10 & abs(tau_raw) < 50;
    dq_valid = dq_raw(valid_idx);  % 清洗后的原始角速度
    tau_valid = tau_raw(valid_idx);  % 清洗后的原始扭矩
    
    % 验证有效数据量（确保参数辨识可靠性）
    fprintf('关节%d原始数据清洗结果：有效数据点数 %d / 总原始数据点数 %d\n', ...
        i, sum(valid_idx), length(dq_raw));
    if sum(valid_idx) < 10  % 有效数据过少时跳过，避免辨识结果无意义
        fprintf('警告：关节 %d 有效原始数据点过少（<10个），跳过该关节参数辨识\n', i);
        continue;
    end
    
    % 3.3 CV模型参数辨识（最小二乘法，匹配稿件1-37段CV模型数学形式）
    % CV模型：tau = Fc*sign(dq) + fv*dq，构建设计矩阵X = [sign(dq_valid), dq_valid]
    X = [sign(dq_valid), dq_valid];  % 设计矩阵（N×2，N为有效数据点数）
    theta = (X' * X) \ (X' * tau_valid);  % 最小二乘求解参数：theta = [Fc; fv]
    
    % 提取CV模型参数（对应稿件1-37段中的F_C和f_v）
    Fc = theta(1);  % Coulomb摩擦系数（Nm）
    fv = theta(2);  % 粘性摩擦系数（Nm·s/rad）
    
    % 3.4 模型性能评估（基于原始数据计算误差指标，匹配稿件1-131段性能指标PI）
    tau_pred = X * theta;  % 基于CV模型的扭矩预测值（使用原始数据）
    rmse = sqrt(mean((tau_valid - tau_pred).^2));  % 均方根误差（RMS）
    ss_res = sum((tau_valid - tau_pred).^2);  % 残差平方和
    ss_tot = sum((tau_valid - mean(tau_valid)).^2);  % 总平方和
    R_squared = 1 - (ss_res / ss_tot);  % 决定系数（评估模型拟合优度）
    
    % 3.5 存储当前关节的CV模型参数与性能指标
    cv_params(i).joint_id = i;  % 关节编号
    cv_params(i).Fc = Fc;  % Coulomb摩擦系数
    cv_params(i).fv = fv;  % 粘性摩擦系数
    cv_params(i).R_squared = R_squared;  % 决定系数
    cv_params(i).rmse = rmse;  % RMS误差
    cv_params(i).num_samples = sum(valid_idx);  % 有效原始数据点数
    
    % 输出当前关节辨识结果（匹配稿件中模型参数展示形式）
    fprintf('关节 %d CV模型参数辨识完成：\n', i);
    fprintf('  Coulomb摩擦系数 Fc = %.6f Nm\n', Fc);
    fprintf('  粘性摩擦系数 fv = %.6f Nm·s/rad\n', fv);
    fprintf('  模型拟合优度 R² = %.4f\n', R_squared);
    fprintf('  预测RMS误差 = %.6f Nm\n', rmse);
end

%% 4. 绘制各关节原始扭矩与CV模型预测扭矩对比图（仅显示扭矩对比，符合用户需求）
fprintf('\n绘制7关节原始扭矩与CV模型预测扭矩对比图...\n');
figure('Position', [100, 100, 1400, 900]);  % 设置画布大小，适配7关节子图

for i = 1:7
    % 跳过未成功辨识参数的关节
    if ~isfield(cv_params(i), 'joint_id') || isempty(cv_params(i).joint_id)
        subplot(3, 3, i);
        axis off;
        text(0.5, 0.5, sprintf('关节 %d\n无有效辨识结果', i), 'HorizontalAlignment', 'center', 'FontSize', 10);
        continue;
    end
    
    % 4.1 提取当前关节的原始数据并清洗（与参数辨识时一致）
    dq_raw = data.(['dq' num2str(i)]);
    tau_raw = data.(['q' num2str(i) '_tau_J_compensate']);
    valid_idx = ~isnan(dq_raw) & ~isnan(tau_raw) & ...
                abs(dq_raw) < 10 & abs(tau_raw) < 50;
    dq_plot = dq_raw(valid_idx);
    tau_plot = tau_raw(valid_idx);
    
    % 4.2 按角速度排序（使对比曲线更清晰，便于观察模型拟合效果）
    [dq_sorted, sort_idx] = sort(dq_plot);
    tau_sorted = tau_plot(sort_idx);
    
    % 4.3 基于CV模型计算预测扭矩（使用辨识出的参数）
    Fc = cv_params(i).Fc;
    fv = cv_params(i).fv;
    tau_pred = Fc * sign(dq_sorted) + fv * dq_sorted;  % 匹配稿件1-37段CV模型公式
    
    % 4.4 绘制对比图（原始数据点+模型预测线，标注参数与拟合优度）
    subplot(3, 3, i);
    % 原始测量扭矩（蓝色散点）
    plot(dq_sorted, tau_sorted, 'b.', 'MarkerSize', 6, 'DisplayName', '原始测量扭矩');
    hold on;
    % CV模型预测扭矩（红色实线）
    plot(dq_sorted, tau_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'CV模型预测扭矩');
    
    % 图表标注（符合稿件实验可视化规范）
    xlabel('关节角速度 (rad/s)', 'FontSize', 9);
    ylabel('关节扭矩 (Nm)', 'FontSize', 9);
    title(sprintf('关节 %d\nFc=%.4f, fv=%.4f, R²=%.3f', ...
        i, Fc, fv, cv_params(i).R_squared), 'FontSize', 10);
    grid on;
    legend('Location', 'best', 'FontSize', 8);
    hold off;
end

% 4.5 添加CV模型参数总结子图（便于快速查看所有关节参数）
subplot(3, 3, 8);
axis off;
text(0.1, 0.95, '7关节CV模型参数总结（原始数据辨识）', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');
y_pos = 0.85;  % 文本垂直位置初始化
for i = 1:7
    if isfield(cv_params(i), 'joint_id') && ~isempty(cv_params(i).joint_id)
        % 显示每个关节的核心参数
        text(0.1, y_pos, sprintf('关节 %d: Fc=%.4f Nm, fv=%.4f Nm·s/rad', ...
            i, cv_params(i).Fc, cv_params(i).fv), 'FontSize', 10, 'HorizontalAlignment', 'left');
        y_pos = y_pos - 0.1;  % 下移文本位置
        if y_pos < 0.1  % 避免文本超出子图范围
            break;
        end
    end
end

% 总标题（明确标注“无滤波”与“原始数据”，符合脚本核心逻辑）
sgtitle('7关节CV摩擦模型扭矩对比（无巴特沃斯滤波，基于原始数据辨识）', 'FontSize', 14, 'FontWeight', 'bold');

%% 5. 保存CV模型辨识参数到MAT文件（便于后续调用验证，符合稿件实验可复现性要求）
% 生成带时间戳的文件名（避免覆盖，便于追溯）
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
mat_filename = sprintf('cv_params_7joints_rawdata_%s.mat', timestamp);

% 保存参数结构（包含各关节参数与性能指标）
save(mat_filename, 'cv_params');
fprintf('\n7关节CV模型参数已保存至文件：%s\n', fullfile(pwd(), mat_filename));
fprintf('保存内容：各关节Fc、fv系数，R²拟合优度，RMS误差，有效数据点数\n');

%% 6. 定义CV摩擦模型调用函数（便于后续验证，匹配稿件中模型应用场景）
function tau_friction = cv_friction_model(dq, params, joint_id)
    % CV摩擦模型计算函数（基于稿件1-37段公式：tau = Fc*sign(dq) + fv*dq）
    % 输入：dq-关节角速度（rad/s），params-存储CV参数的结构，joint_id-关节编号（1-7）
    % 输出：tau_friction-模型预测的摩擦扭矩（Nm）
    
    % 输入合法性检查
    if joint_id < 1 || joint_id > 7 || ~isfield(params(joint_id), 'joint_id')
        error('cv_friction_model: 无效的关节编号（需1-7）或该关节未完成参数辨识');
    end
    
    % 提取当前关节的CV模型参数
    Fc = params(joint_id).Fc;  % Coulomb摩擦系数
    fv = params(joint_id).fv;  % 粘性摩擦系数
    
    % 计算摩擦扭矩（严格遵循稿件1-37段CV模型数学表达式）
    tau_friction = Fc * sign(dq) + fv * dq;
end

%% 7. 后续参数调用示例（注释形式，指导用户使用保存的参数）
% % 步骤1：加载已保存的CV模型参数
% load('cv_params_7joints_rawdata_2024-xx-xx_HH-MM-SS.mat');  % 替换为实际保存的文件名
% 
% % 步骤2：调用CV模型预测摩擦扭矩（示例：关节1，角速度0.5 rad/s）
% dq_test = 0.5;  % 测试角速度（rad/s）
% joint_id_test = 1;  % 测试关节编号
% tau_pred_test = cv_friction_model(dq_test, cv_params, joint_id_test);
% 
% % 步骤3：输出预测结果
% fprintf('关节%d在角速度 %.2f rad/s 时，CV模型预测摩擦扭矩：%.6f Nm\n', ...
%     joint_id_test, dq_test, tau_pred_test);