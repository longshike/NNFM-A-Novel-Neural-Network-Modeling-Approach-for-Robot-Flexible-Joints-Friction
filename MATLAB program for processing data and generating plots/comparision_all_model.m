%% MATLAB script for multi-file friction model comparison plotting
% This script reads 8 CSV files from two robot platforms, applies zero-phase low-pass filtering,
% and creates two figures with 8 subplots each comparing NNFM with other models in IEEE style

clear; clc; close all;

%% Configuration
% File configuration: {filename, actual_torque_column, joint_number, platform, case_number}
file_config = {
    'franka_case1.csv', 'q6_tau_J_compensate', 6, 'Franka', 1;
    'franka_case2.csv', 'q4_tau_J_compensate', 4, 'Franka', 2;
    'franka_case3.csv', 'q2_tau_J_compensate', 2, 'Franka', 3;
    'franka_case4.csv', 'q7_tau_J_compensate', 7, 'Franka', 4;
    'faao_case1.csv',   'q3_tau_J_compensate', 3, 'FAARO', 1;
    'faao_case2.csv',   'q1_tau_J_compensate', 1, 'FAARO', 2;
    'faao_case3.csv',   'q3_tau_J_compensate', 3, 'FAARO', 3;
    'faao_case4.csv',   'q2_tau_J_compensate', 2, 'FAARO', 4;
};

% Filter parameters
cutoff_freq = 1; % Hz - cutoff frequency for low-pass filter
filter_order = 4;  % Filter order

% Time range for plotting (set these manually)
start_time = 0;    % Start time in seconds
end_time = 45;     % End time in seconds

% Model groups for comparison
classical_models = {'CV', 'Dahl', 'Stribeck', 'LuGre', 'NNFM'};
data_driven_models = {'SVM', 'Light_Transformer', 'PINN', 'NNFM'};

% IEEE style parameters
ieee_linewidth = 1.2;
ieee_fontsize = 10;
ieee_fontname = 'Times New Roman';

% Color scheme for models (distinct colors for good visibility)
colors = [
    0, 0.4470, 0.7410;   % Blue
    0.8500, 0.3250, 0.0980; % Orange
    0.9290, 0.6940, 0.1250; % Yellow
    0.4940, 0.1840, 0.5560; % Purple
    0.4660, 0.6740, 0.1880; % Green
    0.3010, 0.7450, 0.9330; % Light Blue
    0.6350, 0.0780, 0.1840; % Red
    0.25, 0.25, 0.25        % Dark Gray
];

%% Pre-process all files
fprintf('Loading and processing %d files...\n', size(file_config, 1));

% Initialize data storage
file_data = cell(size(file_config, 1), 1);

for file_idx = 1:size(file_config, 1)
    filename = file_config{file_idx, 1};
    torque_column = file_config{file_idx, 2};
    joint_number = file_config{file_idx, 3};
    platform = file_config{file_idx, 4};
    case_number = file_config{file_idx, 5};
    
    fprintf('Processing file %d/%d: %s (Joint %d, %s, Case %d)\n', ...
            file_idx, size(file_config, 1), filename, joint_number, platform, case_number);
    
    if ~exist(filename, 'file')
        fprintf('Warning: File %s not found. Skipping.\n', filename);
        continue;
    end
    
    % Read CSV file
    data = readtable(filename);
    
    % Check if required columns exist
    required_columns = [classical_models, data_driven_models, {torque_column, 'time'}];
    missing_columns = setdiff(required_columns, data.Properties.VariableNames);
    if ~isempty(missing_columns)
        fprintf('Warning: Missing columns in %s: %s. Skipping.\n', filename, strjoin(missing_columns, ', '));
        continue;
    end
    
    % Extract time and calculate sampling frequency
    time = data.time;
    if length(time) > 1
        dt = mean(diff(time));
        fs = 1 / dt; % Sampling frequency
    else
        fs = 1000; % Default sampling frequency
    end
    
    % Apply time range selection
    time_mask = (time >= start_time) & (time <= end_time);
    if sum(time_mask) == 0
        fprintf('Warning: No data found in time range [%.2f, %.2f] for %s. Skipping.\n', ...
                start_time, end_time, filename);
        continue;
    end
    
    % Identify data columns to filter
    columns_to_filter = [classical_models, data_driven_models, {torque_column}];
    
    % Apply filtering to each data column
    filtered_data = data;
    for col_idx = 1:length(columns_to_filter)
        col_name = columns_to_filter{col_idx};
        original_data = data.(col_name);
        
        % Remove NaN values temporarily for filtering
        valid_mask = ~isnan(original_data);
        valid_data = original_data(valid_mask);
        
        if length(valid_data) > 10 % Only filter if we have enough data points
            try
                filtered_valid_data = applyZeroPhaseLowPass(valid_data, fs, cutoff_freq, filter_order);
                % Put filtered data back, preserving NaN positions
                temp_data = original_data;
                temp_data(valid_mask) = filtered_valid_data;
                filtered_data.(col_name) = temp_data;
            catch ME
                fprintf('    Warning: Filtering failed for column %s: %s\n', col_name, ME.message);
                % Keep original data if filtering fails
                filtered_data.(col_name) = original_data;
            end
        else
            % Not enough data points, keep original
            filtered_data.(col_name) = original_data;
        end
    end
    
    % Extract filtered data for selected time range
    time_plot = time(time_mask);
    actual_torque_plot = filtered_data.(torque_column)(time_mask);
    
    % Store processed data
    file_data{file_idx} = struct(...
        'time', time_plot, ...
        'actual_torque', actual_torque_plot, ...
        'filtered_data', filtered_data, ...
        'time_mask', time_mask, ...
        'joint_number', joint_number, ...
        'platform', platform, ...
        'case_number', case_number, ...
        'filename', filename);
end

%% Create Figure 1: Classical Models Comparison
fprintf('\nCreating Figure 1: Classical Models Comparison\n');
figure('Position', [50, 50, 1400, 800], 'Color', 'white');

% Subplot labels
subplot_labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)'};

% Franka platform (top row)
for i = 1:4
    subplot(2, 4, i);
    hold on; % Removed grid on
    
    file_idx = i; % Franka files are first 4
    if isempty(file_data{file_idx})
        continue;
    end
    
    data_struct = file_data{file_idx};
    time_plot = data_struct.time;
    actual_torque = data_struct.actual_torque;
    
    % Plot actual torque
    plot(time_plot, actual_torque, 'k-', 'LineWidth', ieee_linewidth+0.3, ...
         'DisplayName', 'Actual');
    
    % Plot classical models
    for j = 1:length(classical_models)
        model = classical_models{j};
        model_data = data_struct.filtered_data.(model)(data_struct.time_mask);
        color_idx = mod(j-1, size(colors,1)) + 1;
        
        if strcmp(model, 'NNFM')
            % Highlight NNFM with thicker dashed line
            plot(time_plot, model_data, '--', 'Color', colors(color_idx,:), ...
                 'LineWidth', ieee_linewidth+0.5, 'DisplayName', model);
        else
            plot(time_plot, model_data, '-', 'Color', colors(color_idx,:), ...
                 'LineWidth', ieee_linewidth, 'DisplayName', model);
        end
    end
    
    % IEEE style formatting
    xlabel('Time (s)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname);
    ylabel('Torque (Nm)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname);
    
    set(gca, 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'Box', 'on');
    xlim([start_time, end_time]);
    
    % Add subplot label at the bottom (adjusted position to avoid overlap with xlabel)
    text(0.02, -0.18, subplot_labels{i}, 'Units', 'normalized', ...
         'FontSize', ieee_fontsize+2, 'FontWeight', 'bold', ...
         'FontName', ieee_fontname, 'HorizontalAlignment', 'left');
end

% FAARO platform (bottom row)
for i = 1:4
    subplot(2, 4, i+4);
    hold on; % Removed grid on
    
    file_idx = i+4; % FAARO files are last 4
    if isempty(file_data{file_idx})
        continue;
    end
    
    data_struct = file_data{file_idx};
    time_plot = data_struct.time;
    actual_torque = data_struct.actual_torque;
    
    % Plot actual torque
    plot(time_plot, actual_torque, 'k-', 'LineWidth', ieee_linewidth+0.3, ...
         'DisplayName', 'Actual');
    
    % Plot classical models
    for j = 1:length(classical_models)
        model = classical_models{j};
        model_data = data_struct.filtered_data.(model)(data_struct.time_mask);
        color_idx = mod(j-1, size(colors,1)) + 1;
        
        if strcmp(model, 'NNFM')
            % Highlight NNFM with thicker dashed line
            plot(time_plot, model_data, '--', 'Color', colors(color_idx,:), ...
                 'LineWidth', ieee_linewidth+0.5, 'DisplayName', model);
        else
            plot(time_plot, model_data, '-', 'Color', colors(color_idx,:), ...
                 'LineWidth', ieee_linewidth, 'DisplayName', model);
        end
    end
    
    % IEEE style formatting
    xlabel('Time (s)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname);
    ylabel('Torque (Nm)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname);
    
    set(gca, 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'Box', 'on');
    xlim([start_time, end_time]);
    
    % Add subplot label at the bottom (adjusted position to avoid overlap with xlabel)
    text(0.02, -0.18, subplot_labels{i+4}, 'Units', 'normalized', ...
         'FontSize', ieee_fontsize+2, 'FontWeight', 'bold', ...
         'FontName', ieee_fontname, 'HorizontalAlignment', 'left');
end

% Create a single legend for the entire figure at the top
% Get handles from the first subplot
subplot(2, 4, 1);
legend_handles = findobj(gca, 'Type', 'line');
legend_labels = get(legend_handles, 'DisplayName');

% Create a single legend at the top of the figure (one row)
lgd = legend(legend_handles, legend_labels, ...
             'Orientation', 'horizontal', ...
             'NumColumns', 6, ... % Increased to fit all in one row
             'FontSize', ieee_fontsize-1, ... % Slightly smaller to fit in one row
             'FontName', ieee_fontname);
lgd.Position = [0.25, 0.94, 0.5, 0.05];

% Save Figure 1
saveas(gcf, 'classical_models_comparison', 'png');
saveas(gcf, 'classical_models_comparison', 'epsc');
saveas(gcf, 'classical_models_comparison', 'fig');

fprintf('Figure 1 saved as classical_models_comparison.*\n');

%% Create Figure 2: Data-Driven Models Comparison
fprintf('\nCreating Figure 2: Data-Driven Models Comparison\n');
figure('Position', [50, 50, 1400, 800], 'Color', 'white');

% Franka platform (top row)
for i = 1:4
    subplot(2, 4, i);
    hold on; % Removed grid on
    
    file_idx = i; % Franka files are first 4
    if isempty(file_data{file_idx})
        continue;
    end
    
    data_struct = file_data{file_idx};
    time_plot = data_struct.time;
    actual_torque = data_struct.actual_torque;
    
    % Plot actual torque
    plot(time_plot, actual_torque, 'k-', 'LineWidth', ieee_linewidth+0.3, ...
         'DisplayName', 'Actual');
    
    % Plot data-driven models
    for j = 1:length(data_driven_models)
        model = data_driven_models{j};
        model_data = data_struct.filtered_data.(model)(data_struct.time_mask);
        color_idx = mod(j+2, size(colors,1)) + 1; % Different color sequence
        
        if strcmp(model, 'NNFM')
            % Highlight NNFM with thicker dashed line
            plot(time_plot, model_data, '--', 'Color', colors(color_idx,:), ...
                 'LineWidth', ieee_linewidth+0.5, 'DisplayName', model);
        else
            plot(time_plot, model_data, '-', 'Color', colors(color_idx,:), ...
                 'LineWidth', ieee_linewidth, 'DisplayName', model);
        end
    end
    
    % IEEE style formatting
    xlabel('Time (s)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname);
    ylabel('Torque (Nm)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname);
    
    set(gca, 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'Box', 'on');
    xlim([start_time, end_time]);
    
    % Add subplot label at the bottom (adjusted position to avoid overlap with xlabel)
    text(0.02, -0.18, subplot_labels{i}, 'Units', 'normalized', ...
         'FontSize', ieee_fontsize+2, 'FontWeight', 'bold', ...
         'FontName', ieee_fontname, 'HorizontalAlignment', 'left');
end

% FAARO platform (bottom row)
for i = 1:4
    subplot(2, 4, i+4);
    hold on; % Removed grid on
    
    file_idx = i+4; % FAARO files are last 4
    if isempty(file_data{file_idx})
        continue;
    end
    
    data_struct = file_data{file_idx};
    time_plot = data_struct.time;
    actual_torque = data_struct.actual_torque;
    
    % Plot actual torque
    plot(time_plot, actual_torque, 'k-', 'LineWidth', ieee_linewidth+0.3, ...
         'DisplayName', 'Actual');
    
    % Plot data-driven models
    for j = 1:length(data_driven_models)
        model = data_driven_models{j};
        model_data = data_struct.filtered_data.(model)(data_struct.time_mask);
        color_idx = mod(j+2, size(colors,1)) + 1; % Different color sequence
        
        if strcmp(model, 'NNFM')
            % Highlight NNFM with thicker dashed line
            plot(time_plot, model_data, '--', 'Color', colors(color_idx,:), ...
                 'LineWidth', ieee_linewidth+0.5, 'DisplayName', model);
        else
            plot(time_plot, model_data, '-', 'Color', colors(color_idx,:), ...
                 'LineWidth', ieee_linewidth, 'DisplayName', model);
        end
    end
    
    % IEEE style formatting
    xlabel('Time (s)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname);
    ylabel('Torque (Nm)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname);
    
    set(gca, 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'Box', 'on');
    xlim([start_time, end_time]);
    
    % Add subplot label at the bottom (adjusted position to avoid overlap with xlabel)
    text(0.02, -0.18, subplot_labels{i+4}, 'Units', 'normalized', ...
         'FontSize', ieee_fontsize+2, 'FontWeight', 'bold', ...
         'FontName', ieee_fontname, 'HorizontalAlignment', 'left');
end

% Create a single legend for the entire figure at the top
% Get handles from the first subplot
subplot(2, 4, 1);
legend_handles = findobj(gca, 'Type', 'line');
legend_labels = get(legend_handles, 'DisplayName');

% Create a single legend at the top of the figure (one row)
lgd = legend(legend_handles, legend_labels, ...
             'Orientation', 'horizontal', ...
             'NumColumns', 5, ... % Adjusted for data-driven models
             'FontSize', ieee_fontsize, ...
             'FontName', ieee_fontname);
lgd.Position = [0.3, 0.94, 0.4, 0.05];

% Save Figure 2
saveas(gcf, 'data_driven_models_comparison', 'png');
saveas(gcf, 'data_driven_models_comparison', 'epsc');
saveas(gcf, 'data_driven_models_comparison', 'fig');

fprintf('Figure 2 saved as data_driven_models_comparison.*\n');

%% Calculate and display RMS errors for all files and models
fprintf('\n=== Model Performance Summary (RMS Error) ===\n');
all_models = unique([classical_models, data_driven_models]);

for file_idx = 1:size(file_config, 1)
    if isempty(file_data{file_idx})
        continue;
    end
    
    data_struct = file_data{file_idx};
    actual_torque = data_struct.actual_torque;
    
    fprintf('\n%s (Joint %d):\n', data_struct.filename, data_struct.joint_number);
    
    for i = 1:length(all_models)
        model = all_models{i};
        model_data = data_struct.filtered_data.(model)(data_struct.time_mask);
        error = model_data - actual_torque;
        rms_error = sqrt(mean(error.^2));
        fprintf('  %s: %.6f Nm\n', model, rms_error);
    end
end

fprintf('\n=== Processing Complete ===\n');
fprintf('Generated two figures with 8 subplots each.\n');
fprintf('Time range: %.1f to %.1f seconds\n', start_time, end_time);
fprintf('Filter: %d Hz cutoff, zero-phase\n', cutoff_freq);

%% Low-pass filter function
function filtered_data = applyZeroPhaseLowPass(data, fs, cutoff_freq, order)
    % Apply zero-phase low-pass Butterworth filter
    if length(data) < 3
        filtered_data = data; % Not enough data points for filtering
        return;
    end
    
    % Normalize cutoff frequency (Nyquist frequency = fs/2)
    Wn = cutoff_freq / (fs/2);
    
    % Design Butterworth filter
    [b, a] = butter(order, Wn, 'low');
    
    % Apply zero-phase filtering
    filtered_data = filtfilt(b, a, data);
end