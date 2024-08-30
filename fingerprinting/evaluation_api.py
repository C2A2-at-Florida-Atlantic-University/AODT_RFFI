import os
import sys
import traceback
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from dataset_api import DatasetAPI
from extractor_api import ExtractorAPI
from singleton import Singleton
from dataset_preparation import awgn
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

class EvaluationAPI(metaclass=Singleton):

    def __init__(self, rx_ids, data_config, aug_config, model_config, root_dir, matlab_src_dir, matlab_session_id, aug_on):
        self.rx_ids = rx_ids
        self.aug_on = aug_on
        self.data_config = data_config
        self.aug_config = aug_config
        self.model_config = model_config
        self.dataset_api = DatasetAPI(root_dir, matlab_src_dir, matlab_session_id, aug_on)
        self.extractor_api = ExtractorAPI()

    def evaluate_preamble_offset(self, rx_id, frame_start_train, offset_range, 
                                 epoch_idx_enroll=0, epoch_idx_identify=1, 
                                 frame_count_enroll=100, frame_count_identify=100, 
                                 enroll_device_idx = [39, 239, 269, 280, 300, 315, 330, 394, 398],
                                 identify_device_idx = [39, 239, 269, 280, 300, 315, 330, 394, 398],
                                 use_pretrained = False, aug_on = False, apply_noise=False, fig_name="Preamble Offset Evaluation"):
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
            accuracy = self.extractor_api.evaluate(feature_extractor, data_enroll, labels_enroll, data_identify, labels_identify, self.model_config)

            print(f"Accuracy: {round(accuracy*100, 2)}%")

            results[offset] = accuracy

        plt.figure(figsize=(10, 8), dpi=80)
        plt.plot(results.keys(), results.values(), '-')
        plt.xlabel("Sequence offset, IQ samples")
        plt.ylabel("Model evaluation accuracy")
        plt.ylim(0, 1)
        plt.title(fig_name)
        plt.show()







    def generate_grid_node_ids(self):
        ids = {}
        coordinates = {}
        node_i = 0
        for i in np.arange(1, 21):
            for j in np.arange(1, 21):
                ids[str(i) + "-" + str(j)] = node_i
                node_i = node_i + 1
                coordinates[node_i] = (i, j)
        return ids, coordinates

    def render_orbit_grid(self, tx_node_ids, rx_node_ids, tx_node_id_curr):
        _, node_coordinates = self._generate_grid_node_ids()

        tx_node_coordinates = [node_coordinates[item] for item in tx_node_ids]
        rx_node_coordinates = [node_coordinates[item] for item in rx_node_ids]
        tx_node_coordinates_curr = node_coordinates[tx_node_id_curr]

        plt.figure(figsize=(8, 8), dpi=80)
        for i in np.arange(1, 21):
            for j in np.arange(1, 21):
                node = (i, j)

                plt.plot(i, j, '.', color='#D3D3D3')

                if node in tx_node_coordinates:
                    plt.plot(i, j, 'o', markerfacecolor='none', markeredgecolor='grey', markersize=8)

        for rx_device in rx_node_coordinates:
            plt.plot(rx_device[0], rx_device[1], 'o', color='black', markersize=10)

        if tx_node_coordinates_curr:
            plt.plot(tx_node_coordinates_curr[0], tx_node_coordinates_curr[1], '.', color='red')
        plt.xticks(np.arange(2, 21, 2))
        plt.yticks(np.arange(2, 21, 2))
        ax = plt.gca()
        ax.invert_xaxis()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        plt.show()

    def _evaluate_fingerprint_similarity_all(self, dev_range, fingerprints_all, rssis_all, ref_device_idx, ref_epoch_idx, epoch_count, show_heatmap = False):
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

                    # Extract the K RSSI values for device i at epoch j and compute an average value
                    rssis = rssis_all[rx_i][i, j, :]

                    rx_distances.append(np.mean(distances))
                    rx_weights.append(self.dataset_api.rssi_to_weight(np.mean(rssis)))

                # rx_weights = [0.5, 0.5, 0.5, 0.5]
                
                avg_distances[i, j] = sum(d * w for d, w in zip(rx_distances, rx_weights))/len(rx_weights)

        device_distances = np.mean(avg_distances, axis=1)

        if show_heatmap:
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

        device_distances.sort()

        return avg_distances, device_distances[0], device_distances[1]
            
    def _evaluate_fingerprint_similarity(self, dev_range, fingerprints, ref_device_idx, ref_epoch_idx, show_heatmap = False):
        N, M, K, D = fingerprints.shape

        # Extract reference fingerprints at device_idx and epoch_idx
        reference_fingerprints = fingerprints[ref_device_idx, ref_epoch_idx, :, :]

        # Initialize a matrix to store average distances with respect to reference fingerprints
        avg_distances = np.zeros((N, M))

        # Compute average Euclidean distance with respect to reference fingerprints for each device and epoch
        for i in range(N):
            for j in range(M):
                # Extract the K fingerprints for device i at epoch j
                data = fingerprints[i, j, :, :]
                # Compute the Euclidean distances between reference fingerprints and current fingerprints
                distances = cdist(reference_fingerprints, data, 'euclidean')
                # Average the distances
                avg_distance = np.mean(distances)
                avg_distances[i, j] = avg_distance

        device_distances = np.mean(avg_distances, axis=1)

        if show_heatmap:
            # Plot the heatmaps side by side
            _, axes = plt.subplots(1, 2, figsize=(20, 8))

            # Plot the heatmap
            sns.heatmap(avg_distances, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Average Euclidean Distance with Respect to Reference'}, yticklabels=dev_range, ax=axes[0])
            axes[0].set_title(f'Average Euclidean Distance of Fingerprints Across Devices and Epochs\n(Reference: device={dev_range[ref_device_idx]}, epoch={ref_epoch_idx})')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Device')

            # Plot the bar chart
            axes[1].bar(np.arange(avg_distances.shape[0]), device_distances)
            axes[1].set_title(f"Device fingerprint comparison\n(Reference: device={dev_range[ref_device_idx]}, epoch={ref_epoch_idx+1})")
            axes[1].set_xlabel('Device')
            axes[1].set_ylabel('Average Distance')

        device_distances.sort()

        return avg_distances, device_distances[0], device_distances[1]

    def _produce_fingerprints(self, rx_name, node_ids_epoch, epochs_override):
        # Retrieve the relevant model
        feature_extractor = self.models[rx_name]

        # Retrieve information about the dataset: paths to dataset files, node IDs, sampling rate
        _, dataset_epoch_paths, _, _, _, samp_rate = self.dataset_api.load_dataset_info(self.data_config['dataset_name'], rx_name, epochs_override)

        fingerprints = np.zeros(shape=(len(node_ids_epoch), len(dataset_epoch_paths), self.data_config['frame_count_epoch'], self.model_config['fp_len']))
        rssis = np.zeros(shape=(len(node_ids_epoch), len(dataset_epoch_paths), self.data_config['frame_count_epoch']))

        print(len(dataset_epoch_paths))

        # Extract fingerprint for every epoch
        for m in np.arange(len(dataset_epoch_paths)):
            label = []
            try:
                print('.', end='')

                # Load all frames/samples for a given epoch
                data, label, rssi = self.dataset_api.load_raw_dataset(dataset_epoch_paths[m], shuffle=False)

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
                rssi_reshaped = rssi.reshape(len(node_ids_epoch), int(data.shape[0]/len(node_ids_epoch)))

                # Save fingerprints for further analysis
                for n in np.arange(data_fps_reshaped.shape[0]):
                    for k in np.arange(data_fps_reshaped.shape[1]):
                        fingerprints[n, m, k, :] = data_fps_reshaped[n, k, :]
                        rssis[n, m, k] = rssi_reshaped[n, k]
            except Exception as e:
                print(dataset_epoch_paths[m])
                print(e)
                traceback.print_exc()

        return samp_rate, fingerprints, rssis

    def _generate_figure_temporal_stability(self, fp_maps, tx_node_names):
        fp_maps_normalized = MinMaxScaler(feature_range=(0, 1)).fit_transform(fp_maps.reshape(-1, 1))
        fp_maps = fp_maps_normalized.reshape(fp_maps.shape)

        plt.figure(figsize=(10, 8), dpi=100)
        plt.rcParams['font.family'] = 'Serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

        y_ticks_vals = []
        y_ticks_labels = []
        for node_i, fp_map in enumerate(fp_maps):
            tx_node_name = f"Node ID: {tx_node_names[node_i]}"
            tx_node_fp_distances = fp_map[node_i, :]

            y_ticks_vals.append(node_i)
            y_ticks_labels.append(tx_node_name)

            plt.plot(tx_node_fp_distances + node_i, label=tx_node_name, color='black')

        # plt.legend()
        plt.yticks(ticks=y_ticks_vals, labels=y_ticks_labels)

    def run_fingerprinting_single(self, rx_name, node_ids_epoch, epochs_override, show_fp_heatmaps):
        # Produce fingerprints
        samp_rate, fingerprints, _ = self._produce_fingerprints(rx_name, node_ids_epoch, epochs_override)

        # Evaluate similarity of fingerprints
        fp_maps = np.zeros((len(node_ids_epoch), len(node_ids_epoch), fingerprints.shape[1]))
        fp_distances = np.zeros((len(node_ids_epoch), 2))
        
        for device_idx in np.arange(len(node_ids_epoch)):
            fp_dist_map, top1_dist, top2_dist = self._evaluate_fingerprint_similarity(node_ids_epoch, fingerprints, device_idx, ref_epoch_idx=0, show_heatmap=show_fp_heatmaps)
            fp_distances[device_idx, 0] = top1_dist
            fp_distances[device_idx, 1] = top2_dist

            fp_maps[device_idx, :, :] = fp_dist_map

        self._generate_figure_temporal_stability(fp_maps, node_ids_epoch)

        # Prepare title for the plot
        fp_discrimination_figure_title = f"Dataset: {self.data_config['dataset_name']}, RX: {rx_name}, Frames: [{self.data_config['frame_count_train']}/{self.data_config['frame_count_epoch']}], Samples: [{self.data_config['samples_count']} ({int(samp_rate/1e6)} MHz)], Augmentation: {self.aug_on}, Alpha: {self.model_config['alpha']}"

        return fp_distances, fp_discrimination_figure_title

    def run_fingerprinting_all(self, rx_nodes, node_ids_epoch, epochs_override, show_fp_heatmaps):
        # Produce fingerprints for all receivers, using corresponding models
        min_epoch_count = sys.maxsize
        fingerprints_all = []
        rssis_all = []
        for i, rx_node in enumerate(rx_nodes):
            print(f"Generating eval finerprints for {rx_node}...")
            # Samp rate is always the same, so we'll just use the last one
            samp_rate, fingerprints, rssis = self._produce_fingerprints(rx_node, node_ids_epoch, epochs_override)
            fingerprints_all.append(fingerprints)
            rssis_all.append(rssis)

            min_epoch_count = min(min_epoch_count, fingerprints.shape[1])
            
        # Evaluate similarity of fingerprints
        fp_maps = np.zeros((len(node_ids_epoch), len(node_ids_epoch), min_epoch_count))
        fp_distances = np.zeros((len(node_ids_epoch), 2))
        for device_idx in np.arange(len(node_ids_epoch)):
            fp_dist_map, top1_dist, top2_dist = self._evaluate_fingerprint_similarity_all(node_ids_epoch, fingerprints_all, rssis_all, device_idx, ref_epoch_idx=0, epoch_count=min_epoch_count, show_heatmap = show_fp_heatmaps)
            fp_distances[device_idx, 0] = top1_dist
            fp_distances[device_idx, 1] = top2_dist

            fp_maps[device_idx, :, :] = fp_dist_map

        self._generate_figure_temporal_stability(fp_maps, node_ids_epoch)

        # Prepare title for the plot (all settings are taken from the last device's config for now, since they're all almost the same)
        fp_discrimination_figure_title = f"Dataset: {self.data_config['dataset_name']}, RX: ALL, Frames: [{self.data_config['frame_count_train']}/{self.data_config['frame_count_epoch']}], Samples: [{self.data_config['samples_count']} ({int(samp_rate/1e6)} MHz)], Augmentation: {self.aug_on}, Alpha: {self.model_config['alpha']}"

        return fp_distances, fp_discrimination_figure_title

if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")