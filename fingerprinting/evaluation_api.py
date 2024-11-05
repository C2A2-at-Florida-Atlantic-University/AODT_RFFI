import os
import sys
import traceback
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
import utils
from dataset_api import DatasetAPI
from extractor_api import ExtractorAPI
from fingerprinting_api import FingerprintingAPI
from dataset_preparation import ChannelIndSpectrogram
from dataset_preparation import awgn
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class EvaluationAPI():

    def __init__(self, rx_ids, data_config, aug_config, model_config, root_dir, matlab_src_dir, matlab_session_id, aug_on):
        self.rx_ids = rx_ids
        self.aug_on = aug_on
        self.data_config = data_config
        self.aug_config = aug_config
        self.model_config = model_config
        self.dataset_api = DatasetAPI(root_dir, matlab_src_dir, matlab_session_id, aug_on)
        self.extractor_api = ExtractorAPI()
        self.fp_api = FingerprintingAPI(
            rx_ids = rx_ids, 
            data_config=data_config, 
            aug_config=aug_config,
            model_config=model_config, 
            root_dir=root_dir, 
            matlab_src_dir=matlab_src_dir, 
            matlab_session_id=matlab_session_id, 
            aug_on=aug_on)

    def evaluate_preamble_offset(self, rx_id, frame_start_train, offset_range, 
                                 epoch_idx_enroll=0, epoch_idx_identify=1, 
                                 frame_count_enroll=100, frame_count_identify=100, 
                                 enroll_device_idx = DatasetAPI.DATASET_V2_TX_MAX_EPOCHS,
                                 identify_device_idx = DatasetAPI.DATASET_V2_TX_MAX_EPOCHS,
                                 use_pretrained = False, aug_on = False, apply_noise=False, fig_name = None, fig_path=None):
        print("Load the training dataset")
        dataset_train_path, dataset_epoch_paths, model_path, node_ids_train, _, samp_rate = self.dataset_api.load_dataset_info(self.data_config['dataset_name'], rx_id, None)
        if aug_on: data, label, rssi = self.dataset_api.load_augmented_dataset(dataset_train_path, samp_rate, self.aug_config, shuffle=True)
        else: data, label, rssi = self.dataset_api.load_raw_dataset(dataset_train_path, shuffle=True)
        data, label, rssi = self.dataset_api.filter_dataset(data, label, rssi, node_ids_train, np.arange(0, self.data_config['frame_count_train']))
        if apply_noise: data = awgn(data, np.arange(self.aug_config['awgn'][0][0], self.aug_config['awgn'][0][1]))
        data_train = data[:, frame_start_train:frame_start_train+self.data_config['samples_count']]
        if use_pretrained: feature_extractor = self.extractor_api.load(os.path.join(model_path, f"extractor_{rx_id}.keras"))
        else: feature_extractor, _ = self.extractor_api.train(data_train, label, node_ids_train, self.model_config, save_path=None)

        results = {}
        for offset in offset_range:
            print(f"Processing offset {offset}")

            # Load data (two epochs: one to enroll devices, another to identify devices)
            data_enroll, labels_enroll, rssi_enroll = self.dataset_api.load_raw_dataset(dataset_epoch_paths[epoch_idx_enroll], shuffle=True)
            data_identify, labels_identify, rssi_identify = self.dataset_api.load_raw_dataset(dataset_epoch_paths[epoch_idx_identify], shuffle=True)

            data_enroll, labels_enroll, _ = self.dataset_api.filter_dataset(data_enroll, labels_enroll, rssi_enroll, dev_range=enroll_device_idx, pkt_range=np.arange(frame_count_enroll))
            data_identify, labels_identify, _ = self.dataset_api.filter_dataset(data_identify, labels_identify, rssi_identify, dev_range=identify_device_idx, pkt_range=np.arange(frame_count_identify))

            data_enroll = data_enroll[:, offset:self.data_config['samples_count']+offset]
            data_identify = data_identify[:, offset:self.data_config['samples_count']+offset]

            # Evaluate the model using the two epochs
            accuracy = self.evaluate_closed_set_knn(feature_extractor, data_enroll, labels_enroll, data_identify, labels_identify, self.model_config)

            print(f"Accuracy: {round(accuracy*100, 2)}%")

            results[offset] = accuracy

        utils.apply_ieee_style()
        plt.plot(results.keys(), results.values(), '-', label = '')
        plt.xlabel("Sequence offset, IQ samples")
        plt.ylabel("Model evaluation accuracy")
        plt.ylim(0, 1)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend()
        # plt.title(fig_name)
        plt.tight_layout()
        if fig_path: plt.savefig(fig_path, format='eps', bbox_inches='tight', pad_inches=0.1)
        plt.show()

        return results

    def evaluate_loss_function(self, rx_id, loss_functions = ['triplet_loss', 'quadruplet_loss'],
                                 epoch_idx_enroll=0, epoch_idx_identify=1, 
                                 frame_count_enroll=100, frame_count_identify=100, 
                                 enroll_device_idx = DatasetAPI.DATASET_V2_TX_MAX_EPOCHS[0:5],
                                 identify_device_idx = DatasetAPI.DATASET_V2_TX_MAX_EPOCHS,
                                 aug_on = False, apply_noise=False, fig_path=None):

        print("Load the training dataset")
        dataset_train_path, dataset_epoch_paths, _, node_ids_train, _, samp_rate = self.dataset_api.load_dataset_info(self.data_config['dataset_name'], rx_id, None)
        if aug_on: data, label, rssi = self.dataset_api.load_augmented_dataset(dataset_train_path, samp_rate, self.aug_config, shuffle=True)
        else: data, label, rssi = self.dataset_api.load_raw_dataset(dataset_train_path, shuffle=True)
        data, label, rssi = self.dataset_api.filter_dataset(data, label, rssi, node_ids_train, np.arange(0, self.data_config['frame_count_train']))
        if apply_noise: data = awgn(data, np.arange(self.aug_config['awgn'][0][0], self.aug_config['awgn'][0][1]))
        data_train = data[:, 0:0+self.data_config['samples_count']]
    
        for loss_function in loss_functions:
            print(f"Evaluating for {loss_function}")

            custom_model_config = self.model_config.copy()
            if loss_function == 'triplet_loss':
                custom_model_config['loss_type'] = 'triplet_loss'
                custom_model_config['loss_num_neg'] = 1
            elif loss_function == 'quadruplet_loss':
                custom_model_config['loss_type'] = 'quadruplet_loss'
                custom_model_config['loss_num_neg'] = 2
            else:
                print("Unknown loss function.")
                return {}

            feature_extractor, _ = self.extractor_api.train(data_train, label, node_ids_train, custom_model_config, save_path=None)

            # Load data (two epochs: one to enroll devices, another to identify devices)
            data_enroll, labels_enroll, rssi_enroll = self.dataset_api.load_raw_dataset(dataset_epoch_paths[epoch_idx_enroll], shuffle=True)
            data_identify, labels_identify, rssi_identify = self.dataset_api.load_raw_dataset(dataset_epoch_paths[epoch_idx_identify], shuffle=True)

            data_enroll, labels_enroll, _ = self.dataset_api.filter_dataset(data_enroll, labels_enroll, rssi_enroll, dev_range=enroll_device_idx, pkt_range=np.arange(frame_count_enroll))
            data_identify, labels_identify, _ = self.dataset_api.filter_dataset(data_identify, labels_identify, rssi_identify, dev_range=identify_device_idx, pkt_range=np.arange(frame_count_identify))

            data_enroll = data_enroll[:, 0:self.data_config['samples_count']]
            data_identify = data_identify[:, 0:self.data_config['samples_count']]

            if fig_path: full_path = os.path.join(fig_path, f'{loss_function}.eps')

            # Evaluate the model using the two epochs (based on the device composition, perform either closed set or open set evaluation)
            if set(enroll_device_idx) == set(identify_device_idx):
                accuracy = self.evaluate_closed_set_knn(feature_extractor, data_enroll, labels_enroll, data_identify, labels_identify, self.model_config, full_path)
                print(f"Accuracy: {round(accuracy*100, 2)}%")
            else: 
                self.evaluate_open_set_knn(feature_extractor, data_enroll, labels_enroll, data_identify, labels_identify, self.model_config, full_path)

    def render_orbit_grid(self, tx_node_ids_1, tx_node_ids_2, rx_node_ids, tx_node_id_curr):
        _, node_coordinates = utils.generate_grid_node_ids()

        tx_node_coordinates_1 = [node_coordinates[item] for item in tx_node_ids_1]
        tx_node_coordinates_2 = [node_coordinates[item] for item in tx_node_ids_2]
        rx_node_coordinates = [node_coordinates[item] for item in rx_node_ids]
        
        plt.figure(figsize=(8, 8), dpi=80)
        for i in np.arange(1, 21):
            for j in np.arange(1, 21):
                node = (i, j)

                plt.plot(i, j, '.', color='#D3D3D3')

                if node in tx_node_coordinates_1:
                    plt.plot(i, j, 'o', markerfacecolor='none', markeredgecolor='blue', markersize=8)
                if node in tx_node_coordinates_2:
                    plt.plot(i, j, 'o', markerfacecolor='none', markeredgecolor='blue', markersize=8)

        for rx_device in rx_node_coordinates:
            plt.plot(rx_device[0], rx_device[1], 'o', color='black', markersize=10)

        if tx_node_id_curr != -1:
            tx_node_coordinates_curr = node_coordinates[tx_node_id_curr]
            plt.plot(tx_node_coordinates_curr[0], tx_node_coordinates_curr[1], '.', color='red')
        plt.xticks(np.arange(2, 21, 2))
        plt.yticks(np.arange(2, 21, 2))
        ax = plt.gca()
        ax.invert_xaxis()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        plt.show()

    def _evaluate_fingerprint_similarity(self, dev_range, fingerprints_all, rssis_all, ref_device_idx, ref_epoch_idx, epoch_count, fig_full_path):
        # Extract reference FPs for each RX device
        ref_fps = []
        for rx_i in np.arange(len(fingerprints_all)):
            ref_fps.append(fingerprints_all[rx_i][ref_device_idx, ref_epoch_idx, :, :])
        
        # Initialize a matrix to store average distances with respect to reference fingerprints
        avg_distances = np.zeros((len(dev_range), epoch_count))

        # Compute average Euclidean distance with respect to reference fingerprints for each device and epoch
        for i in np.arange(avg_distances.shape[0]):
            for j in np.arange(avg_distances.shape[1]):
                rx_distances = []
                rx_weights = []
                for rx_i in np.arange(len(fingerprints_all)):
                    # Extract the K fingerprints for device i at epoch j
                    data = fingerprints_all[rx_i][i, j, :, :]

                    # Compute the Euclidean distances between reference fingerprints and current fingerprints
                    distances = cdist(ref_fps[rx_i], data, 'euclidean')
                    rx_distances.append(np.mean(distances))

                    # Extract the K RSSI values for device i at epoch j and compute an average value
                    if rssis_all:
                        rssis = rssis_all[rx_i][i, j, :]
                        rx_weights.append(self.dataset_api.rssi_to_weight(np.mean(rssis)))
                    else: 
                        rx_weights = None

                # rx_weights = [0.5, 0.5, 0.5, 0.5]
                
                if rx_weights:
                    avg_distances[i, j] = sum(d * w for d, w in zip(rx_distances, rx_weights))/len(rx_weights)
                else:
                    avg_distances[i, j] = np.mean(rx_distances)

        device_distances = np.mean(avg_distances, axis=1)

        if fig_full_path:
            utils.apply_ieee_style()

            # Plot the heatmaps side by side
            _, axes = plt.subplots(1, 2, figsize=(20, 8))

            # Plot the heatmap
            sns.heatmap(avg_distances, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Average Euclidean Distance with Respect to Reference'}, yticklabels=dev_range, ax=axes[0])
            axes[0].set_title(f'Average Euclidean Distance of Fingerprints Across Devices and Epochs\n(Reference: device={dev_range[ref_device_idx]}, epoch={ref_epoch_idx})')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Device')

            # Plot the bar chart
            axes[1].bar(np.arange(avg_distances.shape[0]), device_distances)
            axes[1].set_title(f"Device fingerprint comparison\n(Reference: device={dev_range[ref_device_idx]}, epoch={ref_epoch_idx+1})")
            axes[1].set_xlabel('Device')
            axes[1].set_ylabel('Average Distance')
            
            if fig_full_path: plt.savefig(fig_full_path, format='eps', bbox_inches='tight', pad_inches=0.1)

            plt.show()
            
        device_distances.sort()

        return avg_distances, device_distances[0], device_distances[1]
            
    def _produce_fingerprints(self, models, rx_name, node_ids_epoch, epochs_override):
        # Retrieve the relevant model
        feature_extractor = models[rx_name]

        # Retrieve information about the dataset: paths to dataset files, node IDs, sampling rate
        _, dataset_epoch_paths, _, _, _, samp_rate = self.dataset_api.load_dataset_info(self.data_config['dataset_name'], rx_name, epochs_override)

        fingerprints = np.zeros(shape=(len(node_ids_epoch), len(dataset_epoch_paths), self.data_config['frame_count_epoch'], self.model_config['fp_len']))
        rssis = np.zeros(shape=(len(node_ids_epoch), len(dataset_epoch_paths), self.data_config['frame_count_epoch']))

        # Extract epoch timestamps and sort the epoch paths in ascending order
        epoch_timestamps = []
        epoch_paths_dict = {}
        for epoch_path in dataset_epoch_paths:
            timestamp = utils.extract_unix_timestamp_ms(epoch_path)
            epoch_timestamps.append(timestamp)
            epoch_paths_dict[timestamp] = epoch_path

        epoch_timestamps = np.sort(epoch_timestamps)

        # Extract fingerprint for every epoch
        epoch_idx = 0

        for m in np.arange(len(epoch_timestamps)):
            timestamp = epoch_timestamps[m]
            epoch_path = epoch_paths_dict[timestamp]
            label = []
            try:
                print('.', end='')

                # Load all frames/samples for a given epoch
                data, label, rssi = self.dataset_api.load_raw_dataset(epoch_path, shuffle=False)

                # Filter the dataset (pick specified nodes & frames)
                data, label, rssi = self.dataset_api.filter_dataset(data, label, rssi, dev_range=node_ids_epoch, pkt_range=np.arange(0, self.data_config['frame_count_epoch']))

                # Filter the dataset (pick only a specified number of samples)
                data = data[:, 0:self.data_config['samples_count']]

                # Add AWGN
                # data = awgn(data, np.arange(aug_config['awgn'][0][0], aug_config['awgn'][0][1]))

                # Extract fingerprints from the trained model
                data_fps = self.extractor_api.run(feature_extractor, data, self.model_config)

                # Reshape the fingerprints for easier retrieval
                data_fps_reshaped = data_fps.reshape(len(node_ids_epoch), int(data.shape[0]/len(node_ids_epoch)), data_fps.shape[1])
                if rssi:
                    rssi_reshaped = rssi.reshape(len(node_ids_epoch), int(data.shape[0]/len(node_ids_epoch)))
                else: 
                    rssi_reshaped = None

                # Save fingerprints for further analysis
                for n in np.arange(data_fps_reshaped.shape[0]):
                    for k in np.arange(data_fps_reshaped.shape[1]):
                        fingerprints[n, epoch_idx, k, :] = data_fps_reshaped[n, k, :]
                        if rssi_reshaped:
                            rssis[n, epoch_idx, k] = rssi_reshaped[n, k]
                        else: 
                            rssis = None

                epoch_idx = epoch_idx + 1
            except Exception as e:
                print(epoch_paths_dict[epoch_timestamps[m]])
                print(e)
                traceback.print_exc()

        return samp_rate, fingerprints, rssis, epoch_timestamps

    def _generate_figure_temporal_stability(self, fp_maps, tx_node_names, epoch_timestamps, fig_full_path):
        utils.apply_ieee_style()

        fp_maps_normalized = MinMaxScaler(feature_range=(0, 1)).fit_transform(fp_maps.reshape(-1, 1))
        fp_maps = fp_maps_normalized.reshape(fp_maps.shape)

        y_ticks_vals = []
        y_ticks_labels = []

        # x_ticks_vals = np.append(np.arange(0, len(epoch_timestamps), 25), len(epoch_timestamps) - 1)
        x_ticks_vals = np.arange(0, len(epoch_timestamps), 20)
        x_ticks_labels = [utils.convert_ms_to_time_label(timestamp) for timestamp in np.array(epoch_timestamps)[x_ticks_vals] - epoch_timestamps[0]]

        for node_i, fp_map in enumerate(fp_maps):
            tx_node_name = f"#{tx_node_names[node_i]}"
            tx_node_fp_distances = fp_map[node_i, :]

            y_ticks_vals.append(node_i)
            y_ticks_labels.append(tx_node_name)

            plt.plot(tx_node_fp_distances + node_i, label=tx_node_name, color='black')

        plt.yticks(ticks=y_ticks_vals, labels=y_ticks_labels)
        plt.xticks(ticks=x_ticks_vals, labels=x_ticks_labels)
        if fig_full_path: plt.savefig(fig_full_path, format='eps', bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def evaluate_temporal_stability(self, models, rx_nodes, node_ids_epoch, epochs_override, fig_path, render_heatmaps=False, render_temp_stability=False, rank_dist_fig_file = False):
        # Produce fingerprints for all receivers, using corresponding models
        min_epoch_count = sys.maxsize
        fingerprints_all = []
        rssis_all = []
        epoch_timestamps = []
        for i, rx_node in enumerate(rx_nodes):
            print(f"Generating eval finerprints for {rx_node}...")
            # Samp rate is always the same, so we'll just use the last one
            _, fingerprints, rssis, epoch_timestamps = self._produce_fingerprints(models, rx_node, node_ids_epoch, epochs_override)
            fingerprints_all.append(fingerprints)
            if rssis: 
                rssis_all.append(rssis)
            else: 
                rssis_all = None

            min_epoch_count = min(min_epoch_count, fingerprints.shape[1])
            
        # Evaluate similarity of fingerprints
        fp_maps = np.zeros((len(node_ids_epoch), len(node_ids_epoch), min_epoch_count))
        fp_distances = np.zeros((len(node_ids_epoch), 2))
        for device_idx in np.arange(len(node_ids_epoch)):
            fp_dist_map, top1_dist, top2_dist = self._evaluate_fingerprint_similarity(
                node_ids_epoch, fingerprints_all, rssis_all, 
                device_idx, ref_epoch_idx=0, epoch_count=min_epoch_count, 
                fig_full_path=os.path.join(fig_path, f'fp_heatmap_{device_idx}.eps') if render_heatmaps else None)
            fp_distances[device_idx, 0] = top1_dist
            fp_distances[device_idx, 1] = top2_dist

            fp_maps[device_idx, :, :] = fp_dist_map

        if render_temp_stability:
            self._generate_figure_temporal_stability(fp_maps, node_ids_epoch, epoch_timestamps, os.path.join(fig_path, 'temporal_stability.eps'))

        # Prepare title for the plot (all settings are taken from the last device's config for now, since they're all almost the same)
        lower_line_max = max(fp_distances[:, 0])
        higher_line_min = min(fp_distances[:, 1])

        fp_threshold = (higher_line_min - lower_line_max) / 2 + lower_line_max
        print(f"Re-identification threshold: {fp_threshold}")

        threshold_gap = higher_line_min - lower_line_max

        # Create a figure with two subplots side by side
        utils.apply_ieee_style()
        plt.figure(figsize=(10, 3))
        plt.plot(fp_distances[:, 0], color='blue')#, label="Device with Similarity Rank 1")
        plt.plot(fp_distances[:, 1], color='red')#, label="Device with Similarity Rank 2")
        plt.plot([0, len(fp_distances)-1], [fp_threshold, fp_threshold], color="black", linestyle="--")#, label="Binary Classification Threshold")
        plt.plot([], [], ' ', label=f"Threshold Gap: {round(threshold_gap, 2)}")
        # plt.ylim(0, 0.8)
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0))
        plt.xticks(ticks=range(len(node_ids_epoch)), labels=[f"#{item}" for item in node_ids_epoch])
        plt.grid()
        if rank_dist_fig_file: 
            plt.savefig(os.path.join(fig_path, rank_dist_fig_file), format='eps', bbox_inches='tight', pad_inches=0.1)
        plt.show()

        return fp_distances

    def evaluate_closed_set_knn(self, model, data_epoch_1, labels_epoch_1, data_epoch_2, labels_epoch_2, k=10, fig_path=None):
        epoch_1_device_ids = set(labels_epoch_1.flatten())
        epoch_2_device_ids = set(labels_epoch_2.flatten())

        if epoch_1_device_ids == epoch_2_device_ids:
            print("Great! Epoch #1 and epoch #2 contain identical sets of device IDs. We can perform closed-set evaluation.")
        else:
            print("The device IDs in Epoch #2 and Epoch #1 must be identical. Cannot proceed.")
            return -1

        # Produce fingerprints for the epoch #1
        fps_epoch_1, spec_epoch_1 = self.extractor_api.run(model, data_epoch_1, self.model_config)

        # Perform the enrollment: fit a KNN classifier based on produced fingerprints
        classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        classifier.fit(fps_epoch_1, np.ravel(labels_epoch_1))

        # Produce fingerprints for the epoch #2
        fps_epoch_2, spec_epoch_2 = self.extractor_api.run(model, data_epoch_2, self.model_config)
        labels_epoch_2_predicted = classifier.predict(fps_epoch_2)
        
        # Get the accuracy
        accuracy = accuracy_score(labels_epoch_2, labels_epoch_2_predicted)
        
        if fig_path:
            device_ids = sorted(list(set(labels_epoch_2.flatten())))

            conf_matrix = confusion_matrix(labels_epoch_2, labels_epoch_2_predicted, labels=device_ids)
            conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
            annotations = np.array([["{:.0f}%".format(value) for value in row] for row in conf_matrix_percent])

            utils.apply_ieee_style()
            plt.figure(figsize=(10, 8), dpi=80)
            sns.heatmap(conf_matrix_percent, annot=annotations, fmt="", cmap='YlGnBu', xticklabels=device_ids, yticklabels=device_ids)
            plt.xlabel('Predicted Device ID')
            plt.ylabel('True Device ID')
            plt.gca().xaxis.tick_top()
            plt.gca().xaxis.set_label_position('top') 

            if fig_path: plt.savefig(fig_path, format='eps', bbox_inches='tight', pad_inches=0.1)

            plt.title(f'Closed-set classification accuracy: {np.round(accuracy * 100, 2)}%')
            plt.tight_layout()
            plt.show()

        return accuracy, labels_epoch_2, labels_epoch_2_predicted
    
    def evaluate_open_set_knn(self, model, data_epoch_1, labels_epoch_1, data_epoch_2, labels_epoch_2, model_config, fig_path=None):
        # Here, we also expect two epochs. But we expect that the number set of devices in epoch #1 will be smaller compared to
        # the set of devices in epoch #2.
        epoch_1_device_ids = set(labels_epoch_1.flatten())
        epoch_2_device_ids = set(labels_epoch_2.flatten())

        if epoch_1_device_ids <= epoch_2_device_ids:
            print("Great! Epoch #2 contains more devices than #1, and #1 is a subset of #2. We can start open-set evaluation.")
        else:
            print("Device IDs in epoch #1 must be a subset of device IDs in epoch #2. Cannot proceed.")
            return -1

        # Produce fingerprints for the epoch #1
        fps_epoch_1 = self.extractor_api.run(model, data_epoch_1, model_config)

        # Perform the enrollment: fit a KNN classifier based on produced fingerprints
        classifier = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
        classifier.fit(fps_epoch_1, np.ravel(labels_epoch_1))

        # Produce fingerprints for the epoch #2
        fps_epoch_2 = self.extractor_api.run(model, data_epoch_2, model_config)

        # Find the nearest 15 neighbors in the RFF database and calculate the distances to them.
        distances, _ = classifier.kneighbors(fps_epoch_2)
        
        # Calculate the average distance to the nearest 15 neighbors.
        detection_score = distances.mean(axis=1)
  
        # Create a mask array which will contain 1 if device is from enrolled list, and 0 if it's new
        true_labels = [1 if item in epoch_1_device_ids else 0 for item in labels_epoch_2.flatten()]

        # Compute receiver operating characteristic (ROC).
        fpr, tpr, _ = roc_curve(true_labels, detection_score, pos_label = 1)

        # Invert false positive and true positive ratios to convert from distances to probabilities
        fpr = 1 - fpr  
        tpr = 1 - tpr

        # Compute EER
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        # Compute AUC
        roc_auc = auc(fpr, tpr)

        if fig_path:
            eer_point = min(zip(fpr, tpr), key=lambda x: abs(x[0] - (1-x[1])))

            utils.apply_ieee_style()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
            plt.plot(eer_point[0], eer_point[1], 'ro', markersize=10, label=f'EER = {eer:.2f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True)
            if fig_path: plt.savefig(fig_path, format='eps', bbox_inches='tight', pad_inches=0.1)
            plt.show()

    def evaluate_open_set_knn_multirx(self, models, rx_ids, data_epochs_1, labels_epochs_1, data_epochs_2, labels_epochs_2, rssis_epoch_2, model_config, fig_path=None, fig_title=None):
        ref_rx = rx_ids[0]

        epoch_1_device_ids = set(labels_epochs_1[ref_rx].flatten())
        epoch_2_device_ids = set(labels_epochs_2[ref_rx].flatten())

        if epoch_1_device_ids <= epoch_2_device_ids:
            print(f"Great! Epoch #2 contains more devices than #1, and #1 is a subset of #2. Running open-set for RX: {rx_ids}")
        else:
            print("Device IDs in epoch #1 must be a subset of device IDs in epoch #2. Cannot proceed.")
            return -1

        # Produce detection scores (aka distances) for each receiver
        detection_scores = {}
        for rx_id in rx_ids:
            # Produce fingerprints for the epoch #1
            fps_epoch_1 = self.extractor_api.run(models[rx_id], data_epochs_1[rx_id], model_config)

            # Perform the enrollment: fit a KNN classifier based on produced fingerprints
            classifier = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
            classifier.fit(fps_epoch_1, np.ravel(labels_epochs_1[rx_id]))

            # Produce fingerprints for the epoch #2
            fps_epoch_2 = self.extractor_api.run(models[rx_id], data_epochs_2[rx_id], model_config)

            # Find the nearest 15 neighbors in the RFF database and calculate the distances to them.
            distances, _ = classifier.kneighbors(fps_epoch_2)
            
            # Calculate the average distance to the nearest 15 neighbors.
            detection_scores[rx_id] = distances.mean(axis=1)

        # Combine the scores using RSSI-based weights
        weighted_scores = np.zeros(detection_scores[ref_rx].shape)

        for i in np.arange(weighted_scores.shape[0]):
            if rssis_epoch_2:
                weighted_scores[i] = sum(detection_scores[rx_id][i] * self.dataset_api.rssi_to_weight(rssis_epoch_2[rx_id][i]) for rx_id in rx_ids)
            else:
                weighted_scores[i] = sum(detection_scores[rx_id][i] for rx_id in rx_ids)

        # Create a mask array which will contain 1 if device is from enrolled list, and 0 if it's new
        true_labels = [1 if item in epoch_1_device_ids else 0 for item in labels_epochs_2[ref_rx].flatten()]

        # Compute receiver operating characteristic (ROC).
        fpr, tpr, _ = roc_curve(true_labels, weighted_scores, pos_label = 1)

        # Invert false positive and true positive ratios to convert from distances to probabilities
        fpr = 1 - fpr  
        tpr = 1 - tpr

        # Compute EER
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        # Compute AUC
        roc_auc = auc(fpr, tpr)

        if fig_path:
            utils.apply_ieee_style()
            eer_point = min(zip(fpr, tpr), key=lambda x: abs(x[0] - (1-x[1])))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
            plt.plot(eer_point[0], eer_point[1], 'ro', markersize=10, label=f'EER = {eer:.2f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic (ROC) Curve for Weighted KNN')
            if fig_title: plt.title(fig_title)
            plt.legend(loc="lower right")
            plt.grid(True)
            if fig_path: plt.savefig(fig_path, format='eps', bbox_inches='tight', pad_inches=0.1)
            plt.show()

    def evaluate_closed_set_multirx(self, rx_ids, epoch_idx_enroll = 0, epochs_idx_identify = [1],
                                    enroll_device_idx = DatasetAPI.DATASET_V2_TX_MAX_EPOCHS,
                                    identify_device_idx = DatasetAPI.DATASET_V2_TX_MAX_EPOCHS,
                                    frame_count_enroll = 10, frame_count_identify = 10,
                                    enroll_threshold = 0, identify_threshold = 0.55, fig_path = None):
        self.fp_api.purge_database()
        self.fp_api.load_models()

        _, grid_node_coordinates = utils.generate_grid_node_ids()

        # Enroll all the devices from epoch #1
        enrolled_device_map = {}
        for device_id in enroll_device_idx:
            try:
                print(f"Enrolling device: {grid_node_coordinates[device_id]}")

                # Retrieve frames (across all receivers) for a given device
                frames_rx_all = {}
                for rx_id in rx_ids:
                    _, dataset_epoch_paths, _, _, _, _ = self.dataset_api.load_dataset_info(self.data_config['dataset_name'], rx_id, None)

                    frames_rx_all[rx_id] = self.dataset_api.load_testing_input(dataset_epoch_paths, epoch_idx=epoch_idx_enroll, device_idx=device_id, frame_count=frame_count_enroll)

                # Enroll the device
                enrolled_device_hash = self.fp_api.new_signal(frames_rx_all, new_device_threshold=enroll_threshold)
                print(enrolled_device_hash)

                enrolled_device_map[device_id] = enrolled_device_hash
            except Exception as e:
                print(f"Failed to enroll device {device_id}: {e}")
                print(traceback.format_exc())

        # Create a mapping from device hash to ID
        hash_to_id = {v['device_hash']: k for k, v in enrolled_device_map.items()}

        # Attempt to identify devices from multiple epochs
        true_labels = []
        pred_labels = []

        for epoch_idx in epochs_idx_identify:
            print(f"Epoch {epoch_idx}")
            for device_id in identify_device_idx:
                try:
                    enrolled_device_hash = enrolled_device_map[device_id]['device_hash']
                    print(f"E{epoch_idx}. Identifying device: {grid_node_coordinates[device_id]}. Expected hash: {enrolled_device_hash}")

                    # Retrieve frames (across all receivers) for a given device
                    frames_rx_all = {}
                    for rx_id in rx_ids:
                        _, dataset_epoch_paths, _, _, _, _ = self.dataset_api.load_dataset_info(self.data_config['dataset_name'], rx_id, None)
                        frames_rx_all[rx_id] =self.dataset_api.load_testing_input(dataset_epoch_paths, epoch_idx=epoch_idx, device_idx=device_id, frame_count=frame_count_identify)

                    # Process new signal
                    reidentification_response = self.fp_api.new_signal(frames_rx_all, new_device_threshold=identify_threshold)
                    print(reidentification_response)

                    true_labels.append(device_id)
                    pred_labels.append(hash_to_id[reidentification_response['device_hash']])
                except Exception as e:
                    print(f"Failed to identify device {device_id}: {e}")
                    print(traceback.format_exc())

        # Create the confusion matrix
        cm = confusion_matrix(np.array(true_labels), np.array(pred_labels))

        # Get the unique device IDs (labels)
        labels = sorted(list(set(true_labels)))

        # Plot the confusion matrix
        utils.apply_ieee_style()
        sns.heatmap(cm, annot=True, cbar=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, square=True)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix with Device IDs')
        plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.tight_layout()
        if fig_path: plt.savefig(fig_path, format='eps', bbox_inches='tight', pad_inches=0.1)
        plt.show()

    def evaluate_open_set_multirx(self, rx_ids, epoch_idx_enroll = 0, epochs_idx_identify = 1,
                                    enroll_device_idx = DatasetAPI.DATASET_V2_TX_MAX_EPOCHS[0:5],
                                    identify_device_idx = DatasetAPI.DATASET_V2_TX_MAX_EPOCHS,
                                    frame_count_enroll = 10, frame_count_identify = 10,
                                    enroll_threshold = 0, identify_threshold = 0.55, fig_path = None):
        self.fp_api.load_models()

        _, grid_node_coordinates = utils.generate_grid_node_ids()

        pred_labels = []
        true_labels = []

        for epoch_idx in epochs_idx_identify:
            print(f"============= EPOCH {epoch_idx} ============================")
            self.fp_api.purge_database()

            # Enroll all the devices from epoch #1
            for device_id in enroll_device_idx:
                print(f"Enrolling device: {grid_node_coordinates[device_id]}")

                # Retrieve frames (across all receivers) for a given device
                frames_rx_all = {}
                for rx_id in rx_ids:
                    _, dataset_epoch_paths, _, _, _, _ = self.dataset_api.load_dataset_info(self.data_config['dataset_name'], rx_id, None)
                    frames_rx_all[rx_id] = self.dataset_api.load_testing_input(dataset_epoch_paths, epoch_idx=epoch_idx_enroll, device_idx=device_id, frame_count=frame_count_enroll)

                # Enroll the device
                self.fp_api.new_signal(frames_rx_all, new_device_threshold=enroll_threshold)
            
            for device_id in identify_device_idx:
                try:
                    print(f"E{epoch_idx}. Identifying a device: {grid_node_coordinates[device_id]}")

                    # Retrieve frames (across all receivers) for a given device
                    frames_rx_all = {}
                    for rx_id in rx_ids:
                        _, dataset_epoch_paths, _, _, _, _ = self.dataset_api.load_dataset_info(self.data_config['dataset_name'], rx_id, None)
                        frames_rx_all[rx_id] = self.dataset_api.load_testing_input(dataset_epoch_paths, epoch_idx=epoch_idx, device_idx=device_id, frame_count=frame_count_identify)

                    # Process new signal
                    identification_device_hash = self.fp_api.new_signal(frames_rx_all, new_device_threshold=identify_threshold)

                    pred_labels.append(identification_device_hash['closest_dist'])
                    true_labels.append(0 if device_id in enroll_device_idx else 1)
                except Exception as e:
                    print(f"Failed to identify device {device_id}: {e}")
                    print(traceback.format_exc())

        print(true_labels)
        print(pred_labels)

        # Compute receiver operating characteristic (ROC).
        fpr, tpr, _ = roc_curve(true_labels, pred_labels)

        # Invert false positive and true positive ratios to convert from distances to probabilities
        fpr = 1 - fpr  
        tpr = 1 - tpr

        # Compute EER
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        # Compute AUC
        roc_auc = auc(fpr, tpr)

        if fig_path:
            plt.figure(figsize=(10, 10), dpi=80)
            # utils.apply_ieee_style()
            eer_point = min(zip(fpr, tpr), key=lambda x: abs(x[0] - (1-x[1])))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
            plt.plot(eer_point[0], eer_point[1], 'ro', markersize=10, label=f'EER = {eer:.2f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # plt.title('Receiver Operating Characteristic (ROC) Curve for Weighted KNN')
            plt.legend(loc="lower right")
            plt.grid(True)
            if fig_path: plt.savefig(fig_path, format='eps', bbox_inches='tight', pad_inches=0.1)
            plt.show()

        return {'true_labels': true_labels, 'pred_labels': pred_labels}

if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")