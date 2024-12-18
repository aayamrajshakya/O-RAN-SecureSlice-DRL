% Define the data
malicious_chance = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

% Slice thresholds for eMBB, MmTC, and URLLC (throughput data for each slice)
throughput_emmb = [19922669.0, 18328855.48, 18767154.198, 19085916.902, ...
                   19324988.93, 19484370.282, 19524215.62, 19603906.296, ...
                   19803132.986, 19922669.0, 19922669.0];
               
throughput_mmtc = [6636305.0, 6636305.0, 6636305.0, 6636305.0, ...
                   6636305.0, 6636305.0, 6636305.0, 6636305.0, ...
                   6636305.0, 6636305.0, 6636305.0];
               
throughput_urllc = [625807.0, 625807.0, 625807.0, 625807.0, ...
                    625807.0, 625807.0, 625807.0, 625807.0, ...
                    625807.0, 625807.0, 625807.0];

% Define satisfaction thresholds for each slice (percentage of required throughput)
threshold_emmb = 19922669.0;  % eMBB
threshold_mmtc = 6636305.0;   % MmTC
threshold_urllc = 625807.0;   % URLLC

% Calculate satisfaction factors for each slice type
satisfaction_emmb = throughput_emmb / threshold_emmb;
satisfaction_mmtc = throughput_mmtc / threshold_mmtc;
satisfaction_urllc = throughput_urllc / threshold_urllc;

% Create a new figure
figure;

% Plot throughput vs malicious chance for each slice type
subplot(2,1,1);  % Plot throughput in the first subplot
plot(malicious_chance, throughput_emmb, 'o-', 'DisplayName', 'eMBB', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(malicious_chance, throughput_mmtc, 's-', 'DisplayName', 'MmTC', 'LineWidth', 2, 'MarkerSize', 6);
plot(malicious_chance, throughput_urllc, '^-', 'DisplayName', 'URLLC', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Malicious Chance (%)');
ylabel('Throughput (Bytes)');
title('Throughput vs Malicious Chance for Different Slices');
legend('Location', 'best');
grid on;
hold off;

% Plot satisfaction factor vs malicious chance for each slice type
subplot(2,1,2);  % Plot satisfaction factor in the second subplot
plot(malicious_chance, satisfaction_emmb, 'o-', 'DisplayName', 'eMBB', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(malicious_chance, satisfaction_mmtc, 's-', 'DisplayName', 'MmTC', 'LineWidth', 2, 'MarkerSize', 6);
plot(malicious_chance, satisfaction_urllc, '^-', 'DisplayName', 'URLLC', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Malicious Chance (%)');
ylabel('Satisfaction Factor');
title('Satisfaction Factor vs Malicious Chance for Different Slices');
legend('Location', 'best');
grid on;
hold off;
