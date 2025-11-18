%% MATLAB script for friction model comparison plotting
% This script reads a CSV file, applies zero-phase low-pass filtering,
% and creates two subplots comparing NNFM with other models in IEEE style

clear; clc; close all;

%% Configuration
filename = 'verify_robot_zhixian_slow_processed.csv'; % Change this to your CSV file name
actual_torque_column = 'q3_tau_J_compensate'; % Change according to joint

% Filter parameters
cutoff_freq = 10; % Hz - cutoff frequency for low-pass filter
filter_order = 4;  % Filter order

% Time range for plotting (set these manually)
start_time = 0;    % Start time in seconds
end_time = 30;     % End time in seconds

% Model groups for comparison
classical_models = {'CV', 'Dahl', 'Stribeck', 'LuGre', 'NNFM'};
data_driven_models = {'SVM', 'Light_Transformer', 'PINN', 'NNFM'};

% IEEE style parameters
ieee_linewidth = 1.5;
ieee_fontsize = 12;
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
    0, 0, 0                 % Black
];

%% Load and process data
fprintf('Loading file: %s\n', filename);

if ~exist(filename, 'file')
    error('File %s not found. Please check the filename.', filename);
end

% Read CSV file
data = readtable(filename);

% Check if required columns exist
required_columns = [classical_models, data_driven_models, {actual_torque_column, 'time'}];
missing_columns = setdiff(required_columns, data.Properties.VariableNames);
if ~isempty(missing_columns)
    error('Missing columns: %s', strjoin(missing_columns, ', '));
end

% Extract time and calculate sampling frequency
time = data.time;
if length(time) > 1
    dt = mean(diff(time));
    fs = 1 / dt; % Sampling frequency
    fprintf('Sampling frequency: %.2f Hz\n', fs);
else
    fs = 1000; % Default sampling frequency
    fprintf('Using default sampling frequency: %.2f Hz\n', fs);
end

% Apply time range selection
time_mask = (time >= start_time) & (time <= end_time);
if sum(time_mask) == 0
    error('No data found in the specified time range [%.2f, %.2f] seconds.', start_time, end_time);
end

fprintf('Selected time range: %.2f to %.2f seconds (%d data points)\n', ...
        start_time, end_time, sum(time_mask));

% Identify data columns to filter (all model columns and actual torque)
columns_to_filter = [classical_models, data_driven_models, {actual_torque_column}];

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
            fprintf('Warning: Filtering failed for column %s: %s\n', col_name, ME.message);
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
actual_torque_plot = filtered_data.(actual_torque_column)(time_mask);

%% Create figure with IEEE style
figure('Position', [100, 100, 1000, 800], 'Color', 'white');

% Subplot 1: Classical models comparison
subplot(2,1,1);
hold on; grid on;

% Plot actual torque
plot(time_plot, actual_torque_plot, 'k-', 'LineWidth', ieee_linewidth+0.5, ...
     'DisplayName', 'Actual Torque');

% Plot classical models
for i = 1:length(classical_models)
    model = classical_models{i};
    model_data = filtered_data.(model)(time_mask);
    color_idx = mod(i-1, size(colors,1)) + 1;
    
    if strcmp(model, 'NNFM')
        % Highlight NNFM with thicker line
        plot(time_plot, model_data, '--', 'Color', colors(color_idx,:), ...
             'LineWidth', ieee_linewidth+1, 'DisplayName', model);
    else
        plot(time_plot, model_data, '-', 'Color', colors(color_idx,:), ...
             'LineWidth', ieee_linewidth, 'DisplayName', model);
    end
end

% IEEE style formatting
xlabel('Time (s)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'Interpreter', 'latex');
ylabel('Torque (Nm)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'Interpreter', 'latex');
title('Comparison with Classical Friction Models', ...
      'FontSize', ieee_fontsize+2, 'FontName', ieee_fontname, 'FontWeight', 'bold', 'Interpreter', 'latex');
legend('show', 'Location', 'best', 'FontSize', ieee_fontsize-2, 'FontName', ieee_fontname, 'Interpreter', 'latex');
set(gca, 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'GridAlpha', 0.3);
xlim([start_time, end_time]);

% Subplot 2: Data-driven models comparison
subplot(2,1,2);
hold on; grid on;

% Plot actual torque
plot(time_plot, actual_torque_plot, 'k-', 'LineWidth', ieee_linewidth+0.5, ...
     'DisplayName', 'Actual Torque');

% Plot data-driven models
for i = 1:length(data_driven_models)
    model = data_driven_models{i};
    model_data = filtered_data.(model)(time_mask);
    color_idx = mod(i+3, size(colors,1)) + 1; % Different color sequence
    
    if strcmp(model, 'NNFM')
        % Highlight NNFM with thicker line
        plot(time_plot, model_data, '--', 'Color', colors(color_idx,:), ...
             'LineWidth', ieee_linewidth+1, 'DisplayName', model);
    else
        plot(time_plot, model_data, '-', 'Color', colors(color_idx,:), ...
             'LineWidth', ieee_linewidth, 'DisplayName', model);
    end
end

% IEEE style formatting
xlabel('Time (s)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'Interpreter', 'latex');
ylabel('Torque (Nm)', 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'Interpreter', 'latex');
title('Comparison with Data-Driven Friction Models', ...
      'FontSize', ieee_fontsize+2, 'FontName', ieee_fontname, 'FontWeight', 'bold', 'Interpreter', 'latex');
legend('show', 'Location', 'best', 'FontSize', ieee_fontsize-2, 'FontName', ieee_fontname, 'Interpreter', 'latex');
set(gca, 'FontSize', ieee_fontsize, 'FontName', ieee_fontname, 'GridAlpha', 0.3);
xlim([start_time, end_time]);

%% Add overall title and annotation
sgtitle('Friction Model Performance Comparison', ...
        'FontSize', ieee_fontsize+4, 'FontName', ieee_fontname, 'FontWeight', 'bold', 'Interpreter', 'latex');

% Add filter information annotation
annotation('textbox', [0.02, 0.02, 0.3, 0.05], 'String', ...
           sprintf('Filter: %d Hz cutoff, zero-phase', cutoff_freq), ...
           'FontSize', ieee_fontsize-2, 'FontName', ieee_fontname, ...
           'EdgeColor', 'none', 'Interpreter', 'latex');

%% Calculate and display RMS errors for all models
fprintf('\n=== Model Performance (RMS Error) ===\n');
all_models = unique([classical_models, data_driven_models]);
rms_errors = zeros(length(all_models), 1);

for i = 1:length(all_models)
    model = all_models{i};
    model_data = filtered_data.(model)(time_mask);
    error = model_data - actual_torque_plot;
    rms_errors(i) = sqrt(mean(error.^2));
    fprintf('%s: %.6f Nm\n', model, rms_errors(i));
end

% Find best performing model
[best_rms, best_idx] = min(rms_errors);
fprintf('\nBest performing model: %s (RMS = %.6f Nm)\n', all_models{best_idx}, best_rms);

%% Save the figure
saveas(gcf, 'friction_model_comparison', 'png');
saveas(gcf, 'friction_model_comparison', 'epsc'); % For IEEE publications
saveas(gcf, 'friction_model_comparison', 'fig'); % MATLAB figure

fprintf('\nFigure saved as:\n');
fprintf('  - friction_model_comparison.png\n');
fprintf('  - friction_model_comparison.epsc (for publications)\n');
fprintf('  - friction_model_comparison.fig (MATLAB figure)\n');

fprintf('\n=== Plotting Complete ===\n');

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