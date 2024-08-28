% This source code is a slightly refactored and simplified
% version of the 802.11 waveform recovery & analysis demo:
% Ref: https://www.mathworks.com/help/wlan/ug/recover-and-analyze-packets-in-802-11-waveform.html

close all; clear; clc;

% X_path = '/home/smazokha2016/Desktop/orbit_processor_temp/tx{node_node1-10}_rx{node_node1-1+rxFreq_2462e6+rxGain_10+capLen_2+rxSampRate_25e6}.dat';
X_path = '/home/smazokha2016/Desktop/orbit_experiment_aug_8/training_2024-08-08_18-37-33/tx{node_node7-10}_rx{node_node1-1+rxFreq_2462e6+rxGain_10+capLen_4+rxSampRate_25e6}.dat';

T = find_tx_frames(X_path, 'CBW20', 25e6, '00:60:b3:ac:2b:69', 400);