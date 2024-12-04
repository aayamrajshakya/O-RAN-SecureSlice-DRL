% MATLAB Script: Plotting CDF and Frequencies for "Dueling" Models

% Directory and file pattern (files starting with "dueling")
filePattern = 'ddqn*_mal_*_percent.csv'; % Adjust the pattern if necessary
files = dir(filePattern);

% Initialize storage for legend entries
legendEntriesCDF = {};
legendEntriesFreq = {};
lineStyles = {'-', '--', ':', '-.'}; % Different styles for CDF curves
colors = lines; % Colormap for distinct colors

% Figure setup
figure('Position', [100, 100, 1000, 600]);

% Left y-axis: CDF
yyaxis left;
hold on;
ylabel('Cumulative Probability (CDF)');

% Right y-axis: Frequencies
yyaxis right;
hold on;
ylabel('Relative Frequencies');

% Process each file
for i = 1:length(files)
    fileName = files(i).name;

    % Extract malicious percentage from the filename
    maliciousPercent = regexp(fileName, '_mal_(\d+)_percent', 'tokens');
    if ~isempty(maliciousPercent)
        maliciousPercent = str2double(maliciousPercent{1});
    else
        maliciousPercent = NaN; % Default if not found
    end

    % Read data
    data = readtable(fileName);

    % Ensure the file contains the required columns
    if all(ismember({'Total_DL_Values', 'CDF', 'Frequencies'}, data.Properties.VariableNames))
        % Extract relevant data
        totalDLValues = data.Total_DL_Values;
        cdfValues = data.CDF;
        frequencies = data.Frequencies;

        % Plot CDF on left y-axis
        yyaxis left;
        plot(totalDLValues, cdfValues, 'LineWidth', 2, ...
            'LineStyle', lineStyles{mod(i-1, length(lineStyles)) + 1}, ...
            'Color', colors(mod(i-1, size(colors, 1)) + 1, :));
        legendEntriesCDF{end+1} = sprintf('CDF - Malicious %d%%', maliciousPercent);

        % Plot Frequencies as points on right y-axis
        yyaxis right;
        scatter(totalDLValues, frequencies, 50, ...
            'MarkerFaceColor', colors(mod(i-1, size(colors, 1)) + 1, :), ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.5); % Scatter plot with points
        legendEntriesFreq{end+1} = sprintf('Freq - Malicious %d%%', maliciousPercent);
    else
        warning('File %s does not have required columns.', fileName);
    end
end

% Finalize plot
xlabel('Total DL Bytes');
title('CDF and Frequencies for DDQN Models with Different Malicious Percentages');
legend([legendEntriesCDF, legendEntriesFreq], 'Location', 'best', 'FontSize', 9);
grid on;
hold off;

disp('Plot generated successfully.');
