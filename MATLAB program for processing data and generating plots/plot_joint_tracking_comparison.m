%% NNFM Joint Tracking Control Error Comparison Analysis
% For TMECH Journal Paper Figure 10 - Joint Space Tracking Control Task Validation
clear; clc; close all;

%% Read Data
% Read data without NNFM compensation
data_no_nnfm = readmatrix('test_data_jointmotion.csv');
time_no_nnfm = data_no_nnfm(:,1);  % Time sequence
error_no_nnfm = data_no_nnfm(:,2:end);  % Joint tracking errors

% Read data with NNFM compensation  
data_with_nnfm = readmatrix('test_data_jointmotion_nnfm1.csv');
time_with_nnfm = data_with_nnfm(:,1);  % Time sequence
error_with_nnfm = data_with_nnfm(:,2:end);  % Joint tracking errors

%% Check data length and unify time axis
fprintf('Data Length Check:\n');
fprintf('Without NNFM data: %d data points\n', length(time_no_nnfm));
fprintf('With NNFM data: %d data points\n', length(time_with_nnfm));

% Take minimum length to ensure vector lengths match
min_length = min(length(time_no_nnfm), length(time_with_nnfm));
time = time_no_nnfm(1:min_length);
error_no_nnfm = error_no_nnfm(1:min_length, :);
error_with_nnfm = error_with_nnfm(1:min_length, :);

fprintf('Unified data length: %d data points\n', min_length);

%% Calculate absolute tracking errors
abs_error_no_nnfm = abs(error_no_nnfm);
abs_error_with_nnfm = abs(error_with_nnfm);

%% Zero-phase filtering
fs = 1000;  % Sampling frequency (Hz), adjust based on actual situation
fc = 20;    % Cutoff frequency (Hz)

% Check if sampling frequency is reasonable, recalculate if time interval is not 1ms
if length(time) > 1
    actual_fs = 1 / (time(2) - time(1));
    if abs(actual_fs - fs) > 100  % If difference > 100Hz, use actual sampling frequency
        fs = actual_fs;
        fprintf('Using actual sampling frequency: %.2f Hz\n', fs);
    end
end

% Design filter
if fc < fs/2  % Ensure cutoff frequency is less than Nyquist frequency
    [b, a] = butter(4, fc/(fs/2));  % 4th order Butterworth filter
else
    fc = fs/2.5;  % Reset cutoff frequency
    [b, a] = butter(4, fc/(fs/2));
    fprintf('Adjusted cutoff frequency to: %.2f Hz\n', fc);
end

% Filter absolute error data for each joint
abs_error_no_nnfm_filt = zeros(size(abs_error_no_nnfm));
abs_error_with_nnfm_filt = zeros(size(abs_error_with_nnfm));

num_joints = size(abs_error_no_nnfm, 2);
fprintf('Processing data for %d joints...\n', num_joints);

for i = 1:num_joints
    % Check for NaN values in data
    if any(isnan(abs_error_no_nnfm(:,i)))
        abs_error_no_nnfm_filt(:,i) = abs_error_no_nnfm(:,i);
        fprintf('Joint %d without NNFM contains NaN, skipping filter\n', i);
    else
        abs_error_no_nnfm_filt(:,i) = filtfilt(b, a, abs_error_no_nnfm(:,i));
    end
    
    if any(isnan(abs_error_with_nnfm(:,i)))
        abs_error_with_nnfm_filt(:,i) = abs_error_with_nnfm(:,i);
        fprintf('Joint %d with NNFM contains NaN, skipping filter\n', i);
    else
        abs_error_with_nnfm_filt(:,i) = filtfilt(b, a, abs_error_with_nnfm(:,i));
    end
end

%% Specify joints 4 and 6 for comparison
joint_selected = [4, 6];

% Calculate RMSE and improvement for selected joints
rmse_no_nnfm = sqrt(mean(abs_error_no_nnfm_filt.^2, 1));
rmse_with_nnfm = sqrt(mean(abs_error_with_nnfm_filt.^2, 1));
improvement_ratio = (rmse_no_nnfm - rmse_with_nnfm) ./ rmse_no_nnfm * 100;

fprintf('\nRMSE Improvement for Selected Joints:\n');
fprintf('Joint\tWithout NNFM RMSE\tWith NNFM RMSE\tImprovement\n');
for i = 1:length(joint_selected)
    joint = joint_selected(i);
    fprintf('Joint %d\t%.6f rad\t%.6f rad\t%.2f%%\n', ...
            joint, rmse_no_nnfm(joint), rmse_with_nnfm(joint), improvement_ratio(joint));
end

%% Set IEEE plotting style
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultLineLineWidth', 1.5);

%% Plot both joints on the same figure with RMS improvement annotations
figure('Position', [100, 100, 900, 500]);

% Define colors and line styles
colors = [0.8, 0.2, 0.2;  % Dark red for Joint 4 without NNFM
          0.2, 0.6, 0.2;  % Dark green for Joint 4 with NNFM
          0.2, 0.2, 0.8;  % Dark blue for Joint 6 without NNFM
          0.6, 0.2, 0.8]; % Purple for Joint 6 with NNFM

line_styles = {'-', '--', '-', '--'};
line_widths = [1.8, 1.8, 1.5, 1.5];

% Plot all four curves
hold on;

% Joint 4 curves
plot(time, abs_error_no_nnfm_filt(:,4), 'Color', colors(1,:), ...
     'LineStyle', line_styles{1}, 'LineWidth', line_widths(1));
plot(time, abs_error_with_nnfm_filt(:,4), 'Color', colors(2,:), ...
     'LineStyle', line_styles{2}, 'LineWidth', line_widths(2));

% Joint 6 curves
plot(time, abs_error_no_nnfm_filt(:,6), 'Color', colors(3,:), ...
     'LineStyle', line_styles{3}, 'LineWidth', line_widths(3));
plot(time, abs_error_with_nnfm_filt(:,6), 'Color', colors(4,:), ...
     'LineStyle', line_styles{4}, 'LineWidth', line_widths(4));

% Add labels and formatting
xlabel('Time (s)', 'FontSize', 14);
ylabel('Absolute Tracking Error (rad)', 'FontSize', 14);
title('Joint Tracking Absolute Error Comparison With and Without NNFM Compensation', ...
      'FontSize', 16, 'FontWeight', 'bold');

% Create custom legend
legend_labels = {...
    sprintf('Joint 4 - Without NNFM (RMS: %.4f rad)', rmse_no_nnfm(4)), ...
    sprintf('Joint 4 - With NNFM (RMS: %.4f rad)', rmse_with_nnfm(4)), ...
    sprintf('Joint 6 - Without NNFM (RMS: %.4f rad)', rmse_no_nnfm(6)), ...
    sprintf('Joint 6 - With NNFM (RMS: %.4f rad)', rmse_with_nnfm(6))};
legend(legend_labels, 'Location', 'northeast', 'FontSize', 11);

grid on;
xlim([time(1), time(end)]);

% Add RMS improvement annotations on the plot
annotation_text = {...
    sprintf('Joint 4 RMS Improvement: %.2f%%', improvement_ratio(4)), ...
    sprintf('Joint 6 RMS Improvement: %.2f%%', improvement_ratio(6))};

% Position annotations in the upper part of the plot
y_lim = ylim;
x_pos = time(1) + 0.7 * (time(end) - time(1));
y_pos1 = y_lim(1) + 0.85 * (y_lim(2) - y_lim(1));
y_pos2 = y_lim(1) + 0.75 * (y_lim(2) - y_lim(1));

text(x_pos, y_pos1, annotation_text{1}, ...
     'FontSize', 12, 'FontWeight', 'bold', ...
     'BackgroundColor', 'white', 'EdgeColor', 'black', ...
     'HorizontalAlignment', 'center');

text(x_pos, y_pos2, annotation_text{2}, ...
     'FontSize', 12, 'FontWeight', 'bold', ...
     'BackgroundColor', 'white', 'EdgeColor', 'black', ...
     'HorizontalAlignment', 'center');

%% Save figures
% Save as editable fig file
savefig('Figure10_JointTrackingAbsoluteErrorComparison.fig');

% Save as high-quality PNG for paper
print('-dpng', '-r300', 'Figure10_JointTrackingAbsoluteErrorComparison.png');

%% Plot RMSE comparison bar chart for selected joints
figure('Position', [100, 100, 600, 400]);
bar_data = [rmse_no_nnfm(joint_selected); rmse_with_nnfm(joint_selected)]';
b = bar(bar_data);
set(gca, 'XTickLabel', {'Joint 4', 'Joint 6'}, 'FontSize', 12);
ylabel('RMSE (rad)', 'FontSize', 14);
legend('Without NNFM', 'With NNFM', 'Location', 'northeast', 'FontSize', 12);
title('Joint Tracking Error RMSE Comparison', 'FontSize', 16, 'FontWeight', 'bold');
grid on;

% Add value labels and improvement percentages on bars
for i = 1:length(joint_selected)
    text(i-0.15, bar_data(i,1)+max(bar_data(:))*0.01, ...
         sprintf('%.4f', bar_data(i,1)), 'FontSize', 11, 'FontWeight', 'bold');
    text(i+0.05, bar_data(i,2)+max(bar_data(:))*0.01, ...
         sprintf('%.4f', bar_data(i,2)), 'FontSize', 11, 'FontWeight', 'bold');
    
    % Add improvement percentage between bars
    text(i, max(bar_data(i,:)) + max(bar_data(:))*0.05, ...
         sprintf('Improvement: %.1f%%', improvement_ratio(joint_selected(i))), ...
         'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

savefig('RMSE_Absolute_Comparison.fig');
print('-dpng', '-r300', 'RMSE_Absolute_Comparison.png');

%% Performance summary
fprintf('\n=== Performance Summary ===\n');
fprintf('Joint 4: RMSE reduced from %.6f rad to %.6f rad, improvement %.2f%%\n', ...
        rmse_no_nnfm(4), rmse_with_nnfm(4), improvement_ratio(4));
fprintf('Joint 6: RMSE reduced from %.6f rad to %.6f rad, improvement %.2f%%\n', ...
        rmse_no_nnfm(6), rmse_with_nnfm(6), improvement_ratio(6));

fprintf('\nPlotting completed! Generated:\n');
fprintf('- Figure10_JointTrackingAbsoluteErrorComparison.fig (editable file)\n');
fprintf('- Figure10_JointTrackingAbsoluteErrorComparison.png (high-quality image)\n');
fprintf('- RMSE_Absolute_Comparison.fig (RMSE comparison chart)\n');
fprintf('- RMSE_Absolute_Comparison.png (RMSE comparison chart)\n');