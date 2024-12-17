% MATLAB Script: Plotting Throughput vs Malicious Chance for all models
% given 600 episodes
% MATLAB Script: Plotting Throughput vs Malicious Chance for "Dueling" Models

% Initialize storage for legend entries
lineStyles = {'-', '-', '-'}; % Different styles for the models
colors = lines(3); % Distinct colors for 3 datasets

% Data: Throughput for different models
throughput_ddqn = [27184781, 25491354.14, 26188647.55, 26255056.446666665, ...
                   26686714.28, 26952349.86166667, 26885940.97, 26985554.31, ...
                   26985554.31, 27051963.206666667, 26919145.413333334];


throughput_dqn = [27184781, 25358536.341666665, 26155443.101666667, 26387874.24, ...
                  26753123.171666667, 26952349.86166667, 26852736.516666666, ...
                  26719918.723333333, 26885940.97, 26852736.516666666, ...
                  27018758.758333333];
throughput_dueling = [27184781, 25690580.83, 26321465.343333334, 26620305.378333334, ...
                      26786327.62, 26786327.62, 26885940.97, 26819532.06833333, ...
                      26885940.97, 26919145.413333334, 27018758.758333333];

% Malicious chance: x-axis values (0% to 100% in steps of 10%)
malicious_chance = 0:10:100;

% Figure setup
figure('Position', [100, 100, 1000, 600]);

% Plot throughput for each model
hold on;
plot(malicious_chance, throughput_ddqn, 'LineStyle', lineStyles{2}, 'Color', colors(1,:), ...
    'LineWidth', 2, 'DisplayName', 'DDQN Model');




plot(malicious_chance, throughput_dqn, 'LineStyle', lineStyles{3}, 'Color', colors(2,:), ...
    'LineWidth', 2, 'DisplayName', 'DQN Model');
plot(malicious_chance, throughput_dueling, 'LineStyle', lineStyles{1}, 'Color', colors(3,:), ...
    'LineWidth', 2, 'DisplayName', 'Dueling Model');

% Add labels, title, and legend
xlabel('Malicious Chance (%)');
ylabel('Average Throughput');
title('Throughput vs Malicious Chance for Dueling, DDQN, and DQN Models');
legend('Location', 'best', 'FontSize', 9);
grid on;

% Finalize
hold off;
disp('Plot generated successfully.');
