% MATLAB Script: Plotting Total DL Values, CDF for Multiple Models with Dynamic Colors

% Read the CSV files
data_ddqn = readtable('ddqn_mal_50_percent.csv');
data_dqn = readtable('dqn_mal_50_percent.csv');
data_dueling = readtable('Dueling_mal_50_percent.csv'); % Replace with your actual file name

% Combine all data into a structure for easier handling
models = {'DDQN', 'DQN','Dueling'};
data = {data_ddqn, data_dqn,data_dueling};

% Colors for different models
colors = lines(length(models));

% Create the plot
figure('Position', [100, 100, 1000, 600]);

% Loop through each model and plot its data
for i = 1:length(models)
    model_data = data{i}; % Get the data for the current model
    totalDLValues = model_data.Total_DL_Values; % Extract Total DL Values
    cdfValues = model_data.CDF; % Extract CDF Values
    
    % Plot the data with a unique color and marker
    plot(totalDLValues, cdfValues, '-', 'LineWidth', 2, ...
        'DisplayName', models{i}, 'Color', colors(i, :));
    hold on;
end

% Customize the axes and labels
xlabel('Total DL Bytes');
ylabel('Cumulative Probability (CDF) with 50% Malicious Chance');
%title('CDF Comparison Across Models');
legend('Location', 'best'); % Automatically picks up 'DisplayName' values

% Add grid
grid on;

disp('Plot generated successfully.');
