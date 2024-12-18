% Define the data
malicious_chance = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

% Slice throughput data (eMBB, MmTC, URLLC) for each malicious chance
throughput_emmb = [19922669.0, 19424602.275, 19623828.965, 19723442.31, ...
                   19823055.655, 19922669.0, 19823055.655, 19673635.6375, ...
                   19823055.655, 19823055.655, 19922669.0];
               
throughput_mmtc = [6636305.0, 6470397.375, 6553351.1875, 6553351.1875, ...
                   6636305.0, 6603123.475, 6619714.2375, 6619714.2375, ...
                   6619714.2375, 6619714.2375, 6619714.2375];
               
throughput_urllc = [625807.0, 603903.755, 619548.93, 621113.4475, ...
                    617984.4125, 622677.965, 624242.4825, 616419.895, ...
                    625807.0, 624242.4825, 621113.4475];

% Threshold throughput for satisfaction calculation
threshold_emmb = 19922669.0;  % eMBB
threshold_mmtc = 6636305.0;   % MmTC
threshold_urllc = 625807.0;   % URLLC

% Calculate satisfaction factors
satisfaction_emmb = throughput_emmb / threshold_emmb;
satisfaction_mmtc = throughput_mmtc / threshold_mmtc;
satisfaction_urllc = throughput_urllc / threshold_urllc;

% Create a new figure
figure;

% Plot throughput vs malicious chance for each slice type
subplot(2,1,1);  % First subplot for throughput
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
subplot(2,1,2);  % Second subplot for satisfaction factors
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
