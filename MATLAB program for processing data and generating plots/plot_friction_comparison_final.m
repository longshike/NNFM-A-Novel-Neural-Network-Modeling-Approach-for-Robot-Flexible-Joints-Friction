function plot_friction_comparison_final()
    % 读取数据
    data = readtable('friction_model_comparison_all_results_filtered.csv');
    
    % 处理异常值 - 将所有Inf替换为NaN
    data.RMS(isinf(data.RMS)) = NaN;
    data.MEA(isinf(data.MEA)) = NaN;
    data.IN(isinf(data.IN)) = NaN;
    
    % 设置IEEE期刊风格
    set(0, 'DefaultAxesFontName', 'Times New Roman');
    set(0, 'DefaultTextFontName', 'Times New Roman');
    set(0, 'DefaultAxesFontSize', 12);
    set(0, 'DefaultTextFontSize', 12);
    
    % ========== 图1: 速度和轨迹适应性分析（美化版，无颜色条） ==========
    figure('Position', [100, 100, 1000, 700], 'Color', 'white');
    
    % 包含所有模型
    all_models = {'CV', 'Dahl', 'Stribeck', 'LuGre', 'SVM', 'Light Transformer', 'PINN', 'NNFM'};
    
    % 准备数据 - 按案例和速度计算平均RMS
    cases = {'case1', 'case2', 'case3', 'case4'};
    case_names = {'Case 1 (Simple)', 'Case 2', 'Case 3', 'Case 4 (Complex)'};
    speeds = {'high', 'middle', 'slow'};
    
    % 计算每个模型在每个案例和速度下的平均RMS
    performance_data = zeros(length(cases), length(all_models), length(speeds));
    
    for c = 1:length(cases)
        for m = 1:length(all_models)
            for s = 1:length(speeds)
                idx = strcmp(data.Case, cases{c}) & ...
                      strcmp(data.Model, all_models{m}) & ...
                      strcmp(data.Speed, speeds{s});
                if any(idx)
                    performance_data(c, m, s) = mean(data.RMS(idx), 'omitnan');
                else
                    performance_data(c, m, s) = NaN;
                end
            end
        end
    end
    
    % 创建网格
    [X, Y] = meshgrid(1:length(cases), 1:length(all_models));
    
    % 定义不同速度的颜色（更专业的配色）
    speed_colors = [
        0.85, 0.33, 0.10;  % 高速 - 深橙色
        0.93, 0.69, 0.13;  % 中速 - 金色
        0.00, 0.45, 0.74   % 低速 - 深蓝色
    ];
    
    % 绘制每个速度的曲面
    hold on;
    for s = 1:length(speeds)
        Z = squeeze(performance_data(:, :, s))';
        
        % 绘制曲面，使用更专业的透明度设置
        surf(X, Y, Z, ...
            'FaceAlpha', 0.8, ...
            'EdgeColor', [0.3, 0.3, 0.3], ...
            'EdgeAlpha', 0.3, ...
            'LineWidth', 0.5, ...
            'FaceColor', speed_colors(s, :));
    end
    
    % 设置坐标轴
    set(gca, 'XTick', 1:length(cases), 'XTickLabel', case_names, 'FontSize', 11);
    set(gca, 'YTick', 1:length(all_models), 'YTickLabel', all_models, 'FontSize', 11);
    zlabel('RMS Error', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Trajectory Complexity', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Friction Models', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 设置视角
    view(40, 25);
    
    % 添加图例说明速度
    h_legend = legend('High Speed', 'Middle Speed', 'Low Speed', ...
           'Location', 'northeastoutside', 'FontSize', 11);
    set(h_legend, 'Box', 'on');
    
    % 设置z轴为对数刻度以更好显示差异
    set(gca, 'ZScale', 'log');
    
    % 美化网格
    grid on;
    set(gca, 'GridAlpha', 0.3);
    set(gca, 'GridColor', [0.5, 0.5, 0.5]);
    
    % 设置背景色
    set(gca, 'Color', [0.98, 0.98, 0.98]);
    
    % 保存为fig文件
    saveas(gcf, 'performance_adaptability_3d.fig');
    fprintf('图1: 性能适应性3D图已保存为fig文件\n');
    
    % ========== 图2: 综合性能对比（确保NNFM数据可见） ==========
    figure('Position', [100, 100, 1200, 600], 'Color', 'white');
    
    % 包含所有模型
    all_models = {'CV', 'Dahl', 'Stribeck', 'LuGre', 'SVM', 'Light Transformer', 'PINN', 'NNFM'};
    
    % 定义颜色方案（更专业的配色）
    model_colors = [
        0.50, 0.50, 0.50;  % CV - 中灰色
        0.65, 0.65, 0.65;  % Dahl - 浅灰色
        0.80, 0.80, 0.80;  % Stribeck - 更浅灰色
        0.90, 0.40, 0.40;  % LuGre - 红色
        0.95, 0.60, 0.20;  % SVM - 橙色
        0.40, 0.75, 0.40;  % Light_Transformer - 绿色
        0.75, 0.50, 0.75;  % PINN - 紫色
        0.10, 0.70, 0.30   % NNFM - 专业绿色
    ];
    
    % 计算每个模型的平均RMS、MEA和IN（确保正确处理NNFM的MEA值）
    avg_rms = zeros(1, length(all_models));
    avg_mea = zeros(1, length(all_models));
    avg_in = zeros(1, length(all_models));
    
    for i = 1:length(all_models)
        model_data = data(strcmp(data.Model, all_models{i}), :);
        
        % 确保正确处理所有值，特别是NNFM的MEA
        rms_vals = model_data.RMS;
        mea_vals = model_data.MEA;
        in_vals = model_data.IN;
        
        % 移除NaN值
        rms_vals = rms_vals(~isnan(rms_vals));
        mea_vals = mea_vals(~isnan(mea_vals));
        in_vals = in_vals(~isnan(in_vals));
        
        % 计算平均值
        if ~isempty(rms_vals)
            avg_rms(i) = mean(rms_vals);
        else
            avg_rms(i) = NaN;
        end
        
        if ~isempty(mea_vals)
            avg_mea(i) = mean(mea_vals);
        else
            avg_mea(i) = NaN;
        end
        
        if ~isempty(in_vals)
            avg_in(i) = mean(in_vals);
        else
            avg_in(i) = NaN;
        end
    end
    
    % 准备数据矩阵 - 每个模型有三个指标
    bar_data = [avg_rms; avg_mea; avg_in]';
    
    % 绘制分组柱状图
    x = 1:length(all_models);
    bar_handles = bar(bar_data, 'grouped', 'BarWidth', 0.85);
    
    % 设置柱子颜色和属性
    for i = 1:length(bar_handles)
        for j = 1:length(all_models)
            bar_handles(i).CData(j, :) = model_colors(j, :);
        end
        bar_handles(i).EdgeColor = [0.2, 0.2, 0.2];
        bar_handles(i).LineWidth = 0.8;
    end
    
    % 设置坐标轴
    set(gca, 'XTickLabel', all_models, 'XTickLabelRotation', 45, 'FontSize', 11);
    ylabel('Error Value', 'FontSize', 12, 'FontWeight', 'bold');
    
    % 设置y轴为对数刻度以更好显示差异
    set(gca, 'YScale', 'log');
    
    % 添加图例
    h_legend2 = legend('RMS', 'MEA', 'IN', ...
           'Location', 'northeastoutside', 'FontSize', 11);
    set(h_legend2, 'Box', 'on');
    
    % 美化网格
    grid on;
    set(gca, 'GridAlpha', 0.3);
    set(gca, 'GridColor', [0.5, 0.5, 0.5]);
    
    % 设置背景色
    set(gca, 'Color', [0.98, 0.98, 0.98]);
    
    % 统一标注所有柱状图的数值 - 确保NNFM数据可见
    for i = 1:length(all_models)
        for j = 1:3
            if ~isnan(bar_data(i, j)) && bar_data(i, j) > 0
                % 根据数值大小选择合适的格式
                if bar_data(i, j) < 0.01
                    text_str = sprintf('%.1e', bar_data(i, j));
                elseif bar_data(i, j) < 0.1
                    text_str = sprintf('%.3f', bar_data(i, j));
                else
                    text_str = sprintf('%.2f', bar_data(i, j));
                end
                
                % 计算文本位置 - 在柱子顶部稍上方
                text_y = bar_data(i, j) * 1.2;
                
                % 对于NNFM和其他小数值，确保文本在柱子外部可见
                if bar_data(i, j) < max(bar_data(:)) * 0.05
                    % 设置最小高度，确保文本在柱子外部
                    min_height = max(bar_data(:)) * 0.01;
                    text_y = max(bar_data(i, j) * 1.5, min_height);
                end
                
                % 计算x位置 - 根据指标类型调整
                text_x = i + (j-2)*0.25;
                
                % 对于NNFM，使用更明显的标注
                if strcmp(all_models{i}, 'NNFM')
                    % 使用黑色粗体，确保可见
                    text_color = [0, 0, 0]; % 黑色
                    font_weight = 'bold';
                    font_size = 9;
                    % 添加背景框提高可读性
                    text(text_x, text_y, text_str, ...
                         'HorizontalAlignment', 'center', 'FontSize', font_size, ...
                         'FontWeight', font_weight, 'Color', text_color, ...
                         'BackgroundColor', [1, 1, 1, 0.7], 'EdgeColor', 'black');
                else
                    % 其他模型使用正常标注
                    text_color = 'black';
                    font_weight = 'normal';
                    font_size = 8;
                    text(text_x, text_y, text_str, ...
                         'HorizontalAlignment', 'center', 'FontSize', font_size, ...
                         'FontWeight', font_weight, 'Color', text_color);
                end
            end
        end
    end
    
    % 添加计算效率对比标注
    text(length(all_models)/2, max(bar_data(:))*0.6, ...
         'Computational Efficiency: NNFM vs Light Transformer = 1:8-10', ...
         'HorizontalAlignment', 'center', 'FontSize', 11, 'FontWeight', 'bold', ...
         'Color', [0.8, 0.2, 0.2], 'BackgroundColor', [1, 1, 1, 0.9]);
    
    % 添加模型分类标注
    annotation('textbox', [0.02, 0.95, 0.2, 0.05], 'String', 'Traditional Models', ...
               'FontSize', 10, 'Color', [0.4, 0.4, 0.4], 'FontWeight', 'bold', ...
               'EdgeColor', 'none', 'BackgroundColor', [1, 1, 1, 0.7]);
    annotation('textbox', [0.02, 0.90, 0.2, 0.05], 'String', 'Machine Learning Models', ...
               'FontSize', 10, 'Color', [0.2, 0.6, 0.2], 'FontWeight', 'bold', ...
               'EdgeColor', 'none', 'BackgroundColor', [1, 1, 1, 0.7]);
    
    % 保存为fig文件
    saveas(gcf, 'comprehensive_performance_comparison.fig');
    fprintf('图2: 综合性能对比图已保存为fig文件\n');
    
    % 输出性能对比表格
    fprintf('\n=== 性能对比表格 ===\n');
    fprintf('%-20s %-15s %-15s %-15s\n', ...
            'Model', 'Avg RMS', 'Avg MEA', 'Avg IN');
    fprintf('%-20s %-15s %-15s %-15s\n', ...
            '-----', '-------', '-------', '------');
    
    for i = 1:length(all_models)
        fprintf('%-20s %-15.6f %-15.6f %-15.6f\n', ...
                all_models{i}, avg_rms(i), avg_mea(i), avg_in(i));
    end
end

