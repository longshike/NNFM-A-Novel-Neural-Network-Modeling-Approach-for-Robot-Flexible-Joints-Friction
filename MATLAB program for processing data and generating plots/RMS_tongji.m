%% MATLAB script for friction model performance evaluation with filtering
% This script reads 12 CSV files, applies zero-phase low-pass filtering,
% and calculates RMS, MEA, and IN error metrics for 8 friction models

clear; clc; close all;

%% Configuration
% Define file names and corresponding joints
file_config = {
    'case1_high.csv',    'q6_tau_J_compensate', 6;
    'case1_middle.csv',  'q6_tau_J_compensate', 6;
    'case1_slow.csv',    'q6_tau_J_compensate', 6;
    'case2_high.csv',    'q4_tau_J_compensate', 4;
    'case2_middle.csv',  'q4_tau_J_compensate', 4;
    'case2_slow.csv',    'q4_tau_J_compensate', 4;
    'case3_high.csv',    'q2_tau_J_compensate', 2;
    'case3_middle.csv',  'q2_tau_J_compensate', 2;
    'case3_slow.csv',    'q2_tau_J_compensate', 2;
    'case4_high.csv',    'q7_tau_J_compensate', 7;
    'case4_middle.csv',  'q7_tau_J_compensate', 7;
    'case4_slow.csv',    'q7_tau_J_compensate', 7;
};

% Define model names (as they appear in CSV files)
model_names = {'CV', 'Dahl', 'Stribeck', 'LuGre', 'SVM', 'Light_Transformer', 'PINN', 'NNFM'};

% Filter parameters
cutoff_freq = 10; % Hz - cutoff frequency for low-pass filter
filter_order = 4;  % Filter order

% Initialize results table
results_table = table();

%% Process each CSV file
for file_idx = 1:size(file_config, 1)
    filename = file_config{file_idx, 1};
    torque_column = file_config{file_idx, 2};
    joint_number = file_config{file_idx, 3};
    
    fprintf('Processing file: %s (Joint %d)\n', filename, joint_number);
    
    try
        % Read CSV file
        if exist(filename, 'file')
            data = readtable(filename);
            
            % Check if torque column exists
            if ~ismember(torque_column, data.Properties.VariableNames)
                fprintf('Warning: Column %s not found in %s. Skipping.\n', torque_column, filename);
                continue;
            end
            
            % Extract time and calculate sampling frequency
            if ismember('time', data.Properties.VariableNames)
                time = data.time;
                if length(time) > 1
                    dt = mean(diff(time));
                    fs = 1 / dt; % Sampling frequency
                    fprintf('  Sampling frequency: %.2f Hz\n', fs);
                else
                    fs = 1000; % Default sampling frequency if cannot determine
                    fprintf('  Using default sampling frequency: %.2f Hz\n', fs);
                end
            else
                % If no time column, assume typical sampling frequency
                fs = 1000; % Hz - typical control frequency
                fprintf('  No time column found, using default sampling frequency: %.2f Hz\n', fs);
            end
            
            % Identify data columns to filter (all except time and non-numeric columns)
            data_columns = data.Properties.VariableNames;
            columns_to_filter = {};
            
            for col_idx = 1:length(data_columns)
                col_name = data_columns{col_idx};
                if ~strcmp(col_name, 'time') && isnumeric(data.(col_name))
                    columns_to_filter{end+1} = col_name;
                end
            end
            
            fprintf('  Applying zero-phase low-pass filter (cutoff: %d Hz) to %d columns\n', ...
                    cutoff_freq, length(columns_to_filter));
            
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
            
            data = filtered_data; % Use filtered data for further processing
            
            % Extract actual torque values from filtered data
            actual_torque = data.(torque_column);
            
            % Extract case and speed from filename
            filename_parts = strsplit(filename, '_');
            case_name = filename_parts{1};
            speed_name = filename_parts{2}(1:end-4); % Remove '.csv'
            
            % Calculate metrics for each model
            for model_idx = 1:length(model_names)
                model_name = model_names{model_idx};
                
                % Check if model column exists
                if ~ismember(model_name, data.Properties.VariableNames)
                    fprintf('Warning: Model %s not found in %s. Skipping.\n', model_name, filename);
                    continue;
                end
                
                % Extract predicted torque values from filtered data
                predicted_torque = data.(model_name);
                
                % Remove any NaN values
                valid_idx = ~isnan(actual_torque) & ~isnan(predicted_torque);
                if sum(valid_idx) == 0
                    fprintf('Warning: No valid data for model %s in %s. Skipping.\n', model_name, filename);
                    continue;
                end
                
                actual_clean = actual_torque(valid_idx);
                predicted_clean = predicted_torque(valid_idx);
                
                % Calculate error
                error = predicted_clean - actual_clean;
                
                % Calculate performance metrics
                RMS = sqrt(mean(error.^2));        % Root Mean Square
                MEA = mean(abs(error));            % Mean Absolute Error
                IN = max(abs(error));              % Infinite Norm (Max Absolute Error)
                
                % Create result row
                result_row = table();
                result_row.Case = {case_name};
                result_row.Speed = {speed_name};
                result_row.Joint = joint_number;
                result_row.Model = {model_name};
                result_row.RMS = RMS;
                result_row.MEA = MEA;
                result_row.IN = IN;
                result_row.FileName = {filename};
                
                % Append to results table
                results_table = [results_table; result_row];
            end
        else
            fprintf('Warning: File %s not found. Skipping.\n', filename);
        end
        
    catch ME
        fprintf('Error processing file %s: %s\n', filename, ME.message);
    end
end

%% Display summary statistics
fprintf('\n=== Summary Statistics (After Filtering) ===\n');
for model_idx = 1:length(model_names)
    model_name = model_names{model_idx};
    model_mask = strcmp(results_table.Model, model_name);
    if sum(model_mask) > 0
        model_data = results_table(model_mask, :);
        
        fprintf('\nModel: %s\n', model_name);
        fprintf('Average RMS: %.6f\n', mean(model_data.RMS));
        fprintf('Average MEA: %.6f\n', mean(model_data.MEA));
        fprintf('Average IN:  %.6f\n', mean(model_data.IN));
    else
        fprintf('\nModel: %s - No data available\n', model_name);
    end
end

%% Calculate Best_RMS_Count for each model
fprintf('\n=== Calculating Best Performance Counts ===\n');

% Initialize Best_RMS_Count for each model
best_rms_counts = zeros(length(model_names), 1);

% Get unique scenarios
unique_combinations = unique(results_table(:, {'Case', 'Speed', 'Joint'}), 'rows');

% For each scenario, find the model with the best RMS and count it
for i = 1:height(unique_combinations)
    case_name = unique_combinations.Case{i};
    speed_name = unique_combinations.Speed{i};
    joint_number = unique_combinations.Joint(i);
    
    % Get all models for this scenario
    scenario_mask = strcmp(results_table.Case, case_name) & ...
                   strcmp(results_table.Speed, speed_name) & ...
                   results_table.Joint == joint_number;
    
    scenario_data = results_table(scenario_mask, :);
    
    if height(scenario_data) > 0
        % Find the best RMS in this scenario
        [min_rms, min_idx] = min(scenario_data.RMS);
        
        % Get the model name with the best RMS
        best_model = scenario_data.Model{min_idx};
        
        % Increment the count for this model
        model_idx = find(strcmp(model_names, best_model));
        if ~isempty(model_idx)
            best_rms_counts(model_idx) = best_rms_counts(model_idx) + 1;
        end
    end
end

%% Create a summary table sorted by RMS performance
fprintf('\n=== Overall Model Performance Ranking (by Average RMS) ===\n');
summary_table = table();
for model_idx = 1:length(model_names)
    model_name = model_names{model_idx};
    model_mask = strcmp(results_table.Model, model_name);
    if sum(model_mask) > 0
        model_data = results_table(model_mask, :);
        
        summary_row = table();
        summary_row.Model = {model_name};
        summary_row.Average_RMS = mean(model_data.RMS);
        summary_row.Std_RMS = std(model_data.RMS);
        summary_row.Average_MEA = mean(model_data.MEA);
        summary_row.Std_MEA = std(model_data.MEA);
        summary_row.Average_IN = mean(model_data.IN);
        summary_row.Std_IN = std(model_data.IN);
        summary_row.Best_RMS_Count = best_rms_counts(model_idx);
        
        summary_table = [summary_table; summary_row];
    end
end

% Sort by average RMS (best performance first)
if height(summary_table) > 0
    summary_table = sortrows(summary_table, 'Average_RMS');
    disp(summary_table);
end

%% Save results to a single CSV file
output_filename = 'friction_model_comparison_all_results_filtered.csv';
writetable(results_table, output_filename);
fprintf('\nAll results saved to: %s\n', output_filename);

%% Display best performing models for each case-speed-joint combination
fprintf('\n=== Best Performing Models by Scenario ===\n');
for i = 1:height(unique_combinations)
    case_name = unique_combinations.Case{i};
    speed_name = unique_combinations.Speed{i};
    joint_number = unique_combinations.Joint(i);
    
    scenario_data = results_table(strcmp(results_table.Case, case_name) & ...
                                strcmp(results_table.Speed, speed_name) & ...
                                results_table.Joint == joint_number, :);
    
    if height(scenario_data) > 0
        [~, best_rms_idx] = min(scenario_data.RMS);
        [~, best_mea_idx] = min(scenario_data.MEA);
        [~, best_in_idx] = min(scenario_data.IN);
        
        fprintf('\n%s - %s - Joint %d:\n', case_name, speed_name, joint_number);
        fprintf('  Best RMS: %s (%.6f)\n', scenario_data.Model{best_rms_idx}, scenario_data.RMS(best_rms_idx));
        fprintf('  Best MEA: %s (%.6f)\n', scenario_data.Model{best_mea_idx}, scenario_data.MEA(best_mea_idx));
        fprintf('  Best IN:  %s (%.6f)\n', scenario_data.Model{best_in_idx}, scenario_data.IN(best_in_idx));
    end
end

fprintf('\n=== Processing Complete ===\n');
fprintf('Results have been saved to: %s\n', output_filename);
fprintf('Total records processed: %d\n', height(results_table));
fprintf('Filter parameters: %d Hz cutoff, %dth order Butterworth\n', cutoff_freq, filter_order);

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