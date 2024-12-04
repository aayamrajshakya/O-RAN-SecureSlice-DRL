% MATLAB Script: Plotting Total DL Values, CDF, and Frequencies

% Read the CSV file
data = readtable('Dueling_mal_100_percent.csv'); % Replace with your actual file name

% Extract the relevant columns
totalDLValues = data.Total_DL_Values;
cdfValues = data.CDF;
frequencies = data.Frequencies;

% Create the plot
figure('Position', [100, 100, 1000, 600]);

% Plot the CDF
plot(totalDLValues, cdfValues, 'b-', 'LineWidth', 2);
hold on;

% Plot the Frequencies
yyaxis right; % Use a secondary y-axis for frequencies
plot(totalDLValues, frequencies, 'r--', 'LineWidth', 2);

% Customize the axes and labels
xlabel('Total DL Bytes');
yyaxis left;
ylabel('Cumulative Probability (CDF)');
yyaxis right;
ylabel('Relative Frequencies');
title('CDF and Frequencies of Total DL Bytes');
legend({'CDF', 'Frequencies'}, 'Location', 'best');

% Add grid
grid on;

disp('Plot generated successfully.');
