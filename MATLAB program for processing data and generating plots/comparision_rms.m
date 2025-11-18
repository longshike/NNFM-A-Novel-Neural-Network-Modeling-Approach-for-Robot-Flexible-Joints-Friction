clear; close all; clc;

%% 各模型摩擦力矩预测误差分析
fprintf('=== 各模型摩擦力矩预测误差分析 ===\n');

%% 数据加载
data_file = 'verify_data_carsimotion_mansu.csv'; % 请替换为实际文件名
if ~exist(data_file, 'file')
    error('数据文件不存在: %s', data_file);
end

fprintf('加载数据文件: %s\n', data_file);
data = readtable(data_file);

% 显示数据基本信息
fprintf('数据基本信息:\n');
fprintf('  总样本数: %d\n', height(data));
fprintf('  数据列数: %d\n', width(data));
fprintf('  数据列名:\n');
for i = 1:width(data)
    fprintf('    %d: %s\n', i, data.Properties.VariableNames{i});
end

%% 提取关键数据列
% 根据您描述的列顺序
time = data.time;
q6 = data.q6;
dq6 = data.dq6;
ddq6 = data.ddq6;
actual_torque = data.q6_tau_J_compensate; % 真实扭矩值（参考值）

% 各模型预测值
model_names = {'CV', 'Dahl', 'Stribeck', 'LuGre', 'SVM', 'Light_Transformer', 'PINN', 'NNFM'};
model_predictions = zeros(length(actual_torque), length(model_names));

for i = 1:length(model_names)
    model_predictions(:, i) = data.(model_names{i});
end

fprintf('\n数据统计信息:\n');
fprintf('  实际扭矩范围: [%.6f, %.6f] Nm\n', min(actual_torque), max(actual_torque));
fprintf('  关节6角速度范围: [%.6f, %.6f] rad/s\n', min(dq6), max(dq6));

%% 计算各模型的误差指标
fprintf('\n=== 计算各模型误差指标 ===\n');

% 初始化结果存储
results = table();
results.Model = model_names';
results.RMSE = zeros(length(model_names), 1);
results.MAE = zeros(length(model_names), 1);
results.MaxError = zeros(length(model_names), 1);
results.R2 = zeros(length(model_names), 1);
results.NRMSE = zeros(length(model_names), 1); % 归一化RMSE

for i = 1:length(model_names)
    % 当前模型预测值
    pred = model_predictions(:, i);
    
    % 计算误差
    errors = pred - actual_torque;
    
    % 计算各项指标
    rmse = sqrt(mean(errors.^2));
    mae = mean(abs(errors));
    max_error = max(abs(errors));
    
    % 计算R²
    ss_res = sum(errors.^2);
    ss_tot = sum((actual_torque - mean(actual_torque)).^2);
    r2 = 1 - (ss_res / ss_tot);
    
    % 计算归一化RMSE（相对于实际扭矩范围）
    torque_range = max(actual_torque) - min(actual_torque);
    nrmse = rmse / torque_range;
    
    % 存储结果
    results.RMSE(i) = rmse;
    results.MAE(i) = mae;
    results.MaxError(i) = max_error;
    results.R2(i) = r2;
    results.NRMSE(i) = nrmse;
    
    fprintf('  %-18s: RMSE = %8.6f, MAE = %8.6f, R² = %8.4f\n', ...
        model_names{i}, rmse, mae, r2);
end

%% 按RMSE排序显示结果
fprintf('\n=== 模型性能排名 (按RMSE升序) ===\n');

[~, sorted_idx] = sort(results.RMSE);
sorted_results = results(sorted_idx, :);

for i = 1:height(sorted_results)
    fprintf('  %2d. %-18s: RMSE = %8.6f Nm, MAE = %8.6f Nm, R² = %8.4f\n', ...
        i, sorted_results.Model{i}, sorted_results.RMSE(i), ...
        sorted_results.MAE(i), sorted_results.R2(i));
end

%% 生成详细误差分析报告
fprintf('\n=== 生成详细误差分析报告 ===\n');

% 创建分析文件夹
if ~exist('error_analysis_results', 'dir')
    mkdir('error_analysis_results');
end

% 保存误差结果表格
error_results_file = 'error_analysis_results/model_error_metrics.csv';
writetable(results, error_results_file);
fprintf('误差指标已保存: %s\n', error_results_file);

% 保存排序结果
sorted_results_file = 'error_analysis_results/model_ranking_by_rmse.csv';
writetable(sorted_results, sorted_results_file);
fprintf('模型排名已保存: %s\n', sorted_results_file);

%% 保存各模型的误差数据（便于进一步分析）
fprintf('\n=== 保存详细误差数据 ===\n');

% 创建误差数据表格
error_data_table = table(time, actual_torque);
for i = 1:length(model_names)
    error_col = model_predictions(:, i) - actual_torque;
    error_data_table.(['error_' model_names{i}]) = error_col;
end

error_data_file = 'error_analysis_results/all_models_error_data.csv';
writetable(error_data_table, error_data_file);
fprintf('详细误差数据已保存: %s\n', error_data_file);

%% 可视化结果
fprintf('\n=== 生成可视化结果 ===\n');

% 1. RMSE比较柱状图
figure('Position', [100, 100, 1400, 900]);

subplot(2, 3, 1);
bar(results.RMSE);
set(gca, 'XTickLabel', results.Model, 'XTickLabelRotation', 45);
ylabel('RMSE (Nm)');
title('各模型RMSE比较');
grid on;

% 添加数值标签
for i = 1:length(results.RMSE)
    text(i, results.RMSE(i), sprintf('%.4f', results.RMSE(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% 2. R²比较柱状图
subplot(2, 3, 2);
bar(results.R2);
set(gca, 'XTickLabel', results.Model, 'XTickLabelRotation', 45);
ylabel('R²');
title('各模型R²比较');
ylim([0, 1]);
grid on;

% 添加数值标签
for i = 1:length(results.R2)
    text(i, results.R2(i), sprintf('%.4f', results.R2(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% 3. 预测值与实际值对比（最佳和最差模型）
[~, best_idx] = min(results.RMSE);
[~, worst_idx] = max(results.RMSE);

subplot(2, 3, 3);
plot(time, actual_torque, 'k-', 'LineWidth', 2, 'DisplayName', '实际扭矩');
hold on;
plot(time, model_predictions(:, best_idx), 'g-', 'LineWidth', 1, ...
    'DisplayName', sprintf('最佳: %s', model_names{best_idx}));
plot(time, model_predictions(:, worst_idx), 'r-', 'LineWidth', 1, ...
    'DisplayName', sprintf('最差: %s', model_names{worst_idx}));
xlabel('时间 (s)');
ylabel('扭矩 (Nm)');
title('最佳和最差模型预测对比');
legend('show', 'Location', 'best');
grid on;

% 4. 误差分布直方图
subplot(2, 3, 4);
colors = lines(length(model_names));
for i = 1:length(model_names)
    errors = model_predictions(:, i) - actual_torque;
    histogram(errors, 50, 'FaceColor', colors(i,:), 'FaceAlpha', 0.6, ...
        'DisplayName', model_names{i});
    hold on;
end
xlabel('预测误差 (Nm)');
ylabel('频次');
title('各模型误差分布');
legend('show', 'Location', 'best');
grid on;

% 5. 按速度分段的RMSE分析
subplot(2, 3, 5);
speed_bins = [-inf, -1, -0.1, 0.1, 1, inf]; % 速度分段
speed_labels = {'<-1', '-1~-0.1', '-0.1~0.1', '0.1~1', '>1'};

speed_segmented_rmse = zeros(length(speed_bins)-1, length(model_names));

for bin_idx = 1:(length(speed_bins)-1)
    mask = (dq6 >= speed_bins(bin_idx)) & (dq6 < speed_bins(bin_idx+1));
    for model_idx = 1:length(model_names)
        errors_seg = model_predictions(mask, model_idx) - actual_torque(mask);
        if ~isempty(errors_seg)
            speed_segmented_rmse(bin_idx, model_idx) = sqrt(mean(errors_seg.^2));
        else
            speed_segmented_rmse(bin_idx, model_idx) = NaN;
        end
    end
end

% 绘制热图
imagesc(speed_segmented_rmse);
colorbar;
set(gca, 'XTick', 1:length(model_names), 'XTickLabel', model_names, ...
    'XTickLabelRotation', 45);
set(gca, 'YTick', 1:length(speed_labels), 'YTickLabel', speed_labels);
ylabel('速度分段 (rad/s)');
xlabel('模型');
title('各速度段RMSE热图');

% 6. 模型性能综合评分
subplot(2, 3, 6);
% 综合评分：结合RMSE和R²
normalized_rmse = results.RMSE / max(results.RMSE);
normalized_r2 = 1 - (results.R2 - min(results.R2)) / (max(results.R2) - min(results.R2));
composite_score = 0.7 * (1 - normalized_rmse) + 0.3 * (1 - normalized_r2);

bar(composite_score);
set(gca, 'XTickLabel', results.Model, 'XTickLabelRotation', 45);
ylabel('综合评分 (0-1)');
title('模型综合性能评分');
ylim([0, 1]);
grid on;

% 添加数值标签
for i = 1:length(composite_score)
    text(i, composite_score(i), sprintf('%.3f', composite_score(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

% 添加总标题
sgtitle('关节6摩擦力矩模型性能综合分析', 'FontSize', 16, 'FontWeight', 'bold');

% 保存图像
saveas(gcf, 'error_analysis_results/model_comparison_analysis.png');
fprintf('可视化结果已保存: error_analysis_results/model_comparison_analysis.png\n');

%% 生成文本报告
fprintf('\n=== 生成文本分析报告 ===\n');

report_file = 'error_analysis_results/error_analysis_report.txt';
fid = fopen(report_file, 'w');

fprintf(fid, '关节6摩擦力矩模型误差分析报告\n');
fprintf(fid, '生成时间: %s\n', datestr(now));
fprintf(fid, '数据文件: %s\n', data_file);
fprintf(fid, '样本数量: %d\n\n', height(data));

fprintf(fid, '=== 模型性能排名 (按RMSE) ===\n');
for i = 1:height(sorted_results)
    fprintf(fid, '%2d. %-18s: RMSE = %8.6f Nm, MAE = %8.6f Nm, R² = %8.4f\n', ...
        i, sorted_results.Model{i}, sorted_results.RMSE(i), ...
        sorted_results.MAE(i), sorted_results.R2(i));
end

fprintf(fid, '\n=== 关键发现 ===\n');
fprintf(fid, '最佳模型: %s (RMSE = %.6f Nm)\n', ...
    sorted_results.Model{1}, sorted_results.RMSE(1));
fprintf(fid, '最差模型: %s (RMSE = %.6f Nm)\n', ...
    sorted_results.Model{end}, sorted_results.RMSE(end));
fprintf(fid, 'RMSE范围: %.6f - %.6f Nm\n', ...
    min(results.RMSE), max(results.RMSE));
fprintf(fid, 'R²范围: %.4f - %.4f\n', ...
    min(results.R2), max(results.R2));

fprintf(fid, '\n=== 数据统计 ===\n');
fprintf(fid, '实际扭矩统计:\n');
fprintf(fid, '  最小值: %.6f Nm\n', min(actual_torque));
fprintf(fid, '  最大值: %.6f Nm\n', max(actual_torque));
fprintf(fid, '  平均值: %.6f Nm\n', mean(actual_torque));
fprintf(fid, '  标准差: %.6f Nm\n', std(actual_torque));
fprintf(fid, '关节速度统计:\n');
fprintf(fid, '  最小值: %.6f rad/s\n', min(dq6));
fprintf(fid, '  最大值: %.6f rad/s\n', max(dq6));
fprintf(fid, '  平均值: %.6f rad/s\n', mean(dq6));

fclose(fid);
fprintf('文本分析报告已保存: %s\n', report_file);

%% 输出最终总结
fprintf('\n=== 分析完成总结 ===\n');
fprintf('最佳模型: %s (RMSE = %.6f Nm)\n', ...
    sorted_results.Model{1}, sorted_results.RMSE(1));
fprintf('RMSE改善: 最佳模型比最差模型提升了 %.2f%%\n', ...
    (max(results.RMSE) - min(results.RMSE)) / max(results.RMSE) * 100);

fprintf('\n生成的文件:\n');
fprintf('  误差指标: error_analysis_results/model_error_metrics.csv\n');
fprintf('  模型排名: error_analysis_results/model_ranking_by_rmse.csv\n');
fprintf('  详细误差: error_analysis_results/all_models_error_data.csv\n');
fprintf('  可视化: error_analysis_results/model_comparison_analysis.png\n');
fprintf('  分析报告: error_analysis_results/error_analysis_report.txt\n');

fprintf('\n=== 关节6摩擦力矩模型误差分析完成! ===\n');