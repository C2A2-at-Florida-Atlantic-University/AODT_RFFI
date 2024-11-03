% This source code is a slightly refactored and simplified
% version of the 802.11 waveform recovery & analysis demo:
% Ref: https://www.mathworks.com/help/wlan/ug/recover-and-analyze-packets-in-802-11-waveform.html

close all; clear; clc;

% X_path = '/home/smazokha2016/Desktop/osrbit_processor_temp/tx{node_node1-10}_rx{node_node1-1+rxFreq_2462e6+rxGain_10+capLen_2+rxSampRate_25e6}.dat';
% X_path = '/home/smazokha2016/Desktop/orbit_experiment_aug_8/training_2024-08-08_18-37-33/tx{node_node7-10}_rx{node_node1-1+rxFreq_2462e6+rxGain_10+capLen_4+rxSampRate_25e6}.dat';

% X_path = '/home/smazokha2016/Desktop/orbit_dataset_v3_aug8/raw_dataset_demo/tx{node_node1-11}_rx{node_node1-1+rxFreq_2462e6+rxGain_10+capLen_4+rxSampRate_25e6}.dat';
% X_path = '/home/smazokha2016/Desktop/orbit_dataset_v3_aug8/raw_dataset_demo/tx{node_node2-1}_rx{node_node1-1+rxFreq_2462e6+rxGain_10+capLen_4+rxSampRate_25e6}.dat'
% X_path = '/home/smazokha2016/Desktop/jagannath_dataset_raw/Day2WIFIwb/RASP_PI_4_80211g_OFDM_OTA_TX25_100000000aa23ddd_IQ.dat';
X_path = '/home/smazokha2016/Desktop/wisig_raw/wifi_2021_03_01/node1-1/tx{node:node20-15}_rx{node:node1-1-rxFreq:2462e6-rxGain:0.5-capLen:0.512-rxSampRate:25e6}.dat';

T = find_tx_frames(X_path, 'CBW20', 25e6, '', 400, true);
% T = find_tx_frames(X_path, 'CBW20', 25e6, '00:60:b3:25:bf:f5', 400);
% T = find_tx_frames(X_path, 'CBW20', 20e6, 'na', -1);