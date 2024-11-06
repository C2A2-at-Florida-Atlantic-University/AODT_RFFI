from cProfile import label
from importlib.metadata import distribution
import os
import numpy as np
import h5py
import utils
import matplotlib.pyplot as plt
import cfo_utils
import matlab.engine

np.random.seed(42)

class DatasetAPI():

    DATASET_V1 = 'v1_jul_13' # 25 Msps, 2-hour period, manual capture, only one RX node (node1-1)
    DATASET_V2 = 'v2_jul_19' # 25 Msps, 24-hour period, automated capture, 4 RX nodes
    DATASET_V3 = 'v2_jul_21' # 20 Msps, 4-hour period, automated capture, 4 RX nodes
    DATASET_V4 = 'v3_aug_8'  # 25 Msps, 24-hour period, automated capture, 3 RX nodes (1-1, 1-20, 19-19)
    DATASET_WISIG_OLD = 'wisig_old'
    DATASET_WISIG_NEW = 'wisig_new'
    DATASET_V2V4 = 'v2v4' # here, we combine signals from v2_jul_19 and v3_aug_8 datasets to train multi-day performance

    RX_1 = 'node1-1'
    RX_2 = 'node1-20'
    RX_3 = 'node20-1'
    RX_4 = 'node19-19'

    DATASET_V2_TX_MAX_EPOCHS = [39, 239, 269, 280, 300, 315, 330, 394, 398]
    DATASET_V4_TX_MAX_EPOCHS = [1, 259, 10, 269, 398, 247, 280, 186, 315, 189]
    DATASET_V4_TX_MAX_DEVICES = [1, 259, 10, 394, 269, 270, 398, 273, 280, 300, 186, 315, 189, 330, 209, 219, 247, 379, 252]
    DATASET_WISIG_TX = [239, 242, 266, 269, 280, 300, 315, 329, 330, 360, 378, 380, 391, 394, 398]

    DATASET_V2_EPOCHS_MAX_DEVICES = None
    # DATASET_V4_EPOCHS_MAX_DEVICES = None
    DATASET_V4_EPOCHS_MAX_DEVICES = [ 
        'epoch_2024-08-09_17-44-31',
        'epoch_2024-08-09_09-45-28',
        'epoch_2024-08-09_06-24-29',
        'epoch_2024-08-09_13-14-55',
        'epoch_2024-08-09_09-15-25',
        'epoch_2024-08-09_01-12-19',
        'epoch_2024-08-08_19-59-37',
        'epoch_2024-08-08_20-33-18',
        'epoch_2024-08-09_03-34-38',
        'epoch_2024-08-08_19-19-27',
        'epoch_2024-08-09_04-30-31',
        'epoch_2024-08-09_00-15-50',
        'epoch_2024-08-09_04-59-00',
        'epoch_2024-08-08_23-15-19',
        'epoch_2024-08-09_08-47-08',
        'epoch_2024-08-08_21-40-14',
        'epoch_2024-08-09_11-16-36',
        'epoch_2024-08-08_21-04-40',
        'epoch_2024-08-09_14-40-55',
        'epoch_2024-08-09_03-05-59',
        'epoch_2024-08-09_16-11-55',
        'epoch_2024-08-09_10-46-06',
        'epoch_2024-08-08_22-15-27',
        'epoch_2024-08-09_05-56-24',
        'epoch_2024-08-09_17-14-46',
        'epoch_2024-08-09_10-15-26',
        'epoch_2024-08-09_15-41-02',
        'epoch_2024-08-09_19-53-51',
        'epoch_2024-08-09_19-21-11',
        'epoch_2024-08-09_05-27-49'
        ]


    def __init__(self, root_dir, matlab_src_dir, matlab_session_id, aug_on=False, seed=42):
        self.rng = np.random.RandomState(seed)
        self.root_dir = root_dir
        self.dataset_v1_path = os.path.join(self.root_dir, 'orbit_dataset_v1')
        self.dataset_v2_path = os.path.join(self.root_dir, 'orbit_dataset_v2_jul19')
        self.dataset_v3_path = os.path.join(self.root_dir, 'orbit_dataset_v2_jul21')
        self.dataset_v4_path = os.path.join(self.root_dir, 'orbit_dataset_v3_aug8')
        self.dataset_wisig_old_path = os.path.join(self.root_dir, 'wisig_dataset_old_1rx')
        self.dataset_wisig_new_path = os.path.join(self.root_dir, 'wisig_dataset_new_1rx')

        if aug_on:
            self.mateng = matlab.engine.connect_matlab(matlab_session_id)
            self.mateng.cd(matlab_src_dir, nargout=0)
        else: self.mateng = None

    def _load_dataset_v2v4(self, rx_name):
        dataset_train_path = os.path.join(self.dataset_v2_path, rx_name + '_training_2024-07-20_00-50-38.h5'),
        dataset_epoch_paths = [
            os.path.join(self.dataset_v2_path, rx_name + '_epoch_2024-07-20_01-15-26.h5'), # D1 E1
            os.path.join(self.dataset_v2_path, rx_name + '_epoch_2024-07-20_01-25-24.h5'), # E2
            # os.path.join(self.dataset_v4_path, rx_name + '_epoch_2024-08-08_19-19-27.h5'), # D2 E1
        ]
        model_path = os.path.join(self.dataset_v2_path, 'my_models')
        samp_rate = 25e6
        return dataset_train_path, dataset_epoch_paths, model_path, samp_rate

    def  _load_dataset_wisig_old(self, equalized=False):
        equalized_str = 'eq' if equalized else 'non_eq'
        dataset_train_paths = [
            os.path.join(self.dataset_wisig_old_path, 'wisig_dataset-2021_03_01', 'Train', f'node1-1_{equalized_str}_train.h5'),
            os.path.join(self.dataset_wisig_old_path, 'wisig_dataset-2021_03_08', 'Train', f'node1-1_{equalized_str}_train.h5')
        ]
        dataset_epoch_paths = [
            os.path.join(self.dataset_wisig_old_path, 'wisig_dataset-2021_03_01', 'Test', f'{equalized_str}_epoch_2021-03-01_00-00-00.h5'),
            os.path.join(self.dataset_wisig_old_path, 'wisig_dataset-2021_03_08', 'Test', f'{equalized_str}_epoch_2021-03-08_00-00-00.h5')
        ]
        model_path = os.path.join(self.dataset_wisig_old_path, 'my_models')
        samp_rate = 25e6
        return dataset_train_paths, dataset_epoch_paths, model_path, samp_rate

    def _load_dataset_wisig_new(self, equalized=False):
        equalized_str = 'eq' if equalized else 'non_eq'
        dataset_paths = [
            os.path.join(self.dataset_wisig_new_path, 'wisig_dataset-2021_03_01', f"node1-1_{equalized_str}_wifi_2021_03_01.h5"),
            os.path.join(self.dataset_wisig_new_path, 'wisig_dataset-2021_03_08', f"node1-1_{equalized_str}_wifi_2021_03_08.h5"),
            os.path.join(self.dataset_wisig_new_path, 'wisig_dataset-2021_03_15', f"node1-1_{equalized_str}_wifi_2021_03_15.h5"),
            os.path.join(self.dataset_wisig_new_path, 'wisig_dataset-2021_03_23', f"node1-1_{equalized_str}_wifi_2021_03_23.h5")
        ]
        model_path = os.path.join(self.dataset_wisig_old_path, 'my_models')
        samp_rate = 25e6
        return dataset_paths, model_path, samp_rate

    def _load_dataset_v1(self, rx_name, allowed_epochs):
        dataset_train_path = os.path.join(self.dataset_v1_path, 'training_2024-07-13_06-53-20', rx_name + '_non_eq_train.h5')
        model_path = os.path.join(self.dataset_v1_path, 'my_models')
        dataset_epoch_paths = []
        for f in os.listdir(self.dataset_v1_path):
            if not f.startswith('epoch_'): continue

            if allowed_epochs and not any(allowed_epoch in f for allowed_epoch in allowed_epochs): continue
                
            dataset_epoch_paths.append(os.path.join(self.dataset_v1_path, f, rx_name + '_non_eq_test.h5'))

        samp_rate = 25e6

        return dataset_train_path, dataset_epoch_paths, model_path, samp_rate

    def _load_dataset_v2(self, rx_name, allowed_epochs):
        dataset_train_path = os.path.join(self.dataset_v2_path, rx_name + '_training_2024-07-20_00-50-38.h5')
        model_path = os.path.join(self.dataset_v2_path, 'my_models')
        dataset_epoch_paths = []
        for f in os.listdir(self.dataset_v2_path):
            if not f.startswith(rx_name + '_epoch'): continue

            if allowed_epochs and not any(allowed_epoch in f for allowed_epoch in allowed_epochs): continue

            dataset_epoch_paths.append(os.path.join(self.dataset_v2_path, f))
        
        samp_rate = 25e6

        return dataset_train_path, dataset_epoch_paths, model_path, samp_rate

    def _load_dataset_v3(self, rx_name, allowed_epochs):
        dataset_train_path = os.path.join(self.dataset_v3_path, rx_name + '_training_2024-07-21_14-49-09.h5')
        model_path = os.path.join(self.dataset_v3_path, 'my_models')
        dataset_epoch_paths = []
        for f in os.listdir(self.dataset_v3_path):
            if not f.startswith(rx_name + '_epoch'): continue

            if allowed_epochs and not any(allowed_epoch in f for allowed_epoch in allowed_epochs): continue

            dataset_epoch_paths.append(os.path.join(self.dataset_v3_path, f))

        samp_rate = 20e6

        return dataset_train_path, dataset_epoch_paths, model_path, samp_rate

    def _load_dataset_v4(self, rx_name, allowed_epochs):
        dataset_train_path = os.path.join(self.dataset_v4_path, rx_name + '_training_2024-08-08_18-37-33.h5')
        model_path = os.path.join(self.dataset_v4_path, 'my_models')
        dataset_epoch_paths = []
        for f in os.listdir(self.dataset_v4_path):
            if not f.startswith(rx_name + '_epoch'): continue

            if allowed_epochs and not any(allowed_epoch in f for allowed_epoch in allowed_epochs): continue

            dataset_epoch_paths.append(os.path.join(self.dataset_v4_path, f))

        samp_rate = 25e6

        return dataset_train_path, dataset_epoch_paths, model_path, samp_rate

    def _get_dataset_devices(self, labels, show=False):
        # Retrives a list of distinct node IDs from a given array of nodes (sorted)
        devices = list(set(labels.flatten()))
        devices.sort()

        if show: print(devices)

        return devices

    def load_dataset_info(self, dataset_name, rx_name, allowed_epochs, wisig_equalized=False, wisig_disjoint=False):
        if dataset_name == self.DATASET_V1:
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_v1(rx_name, allowed_epochs)
        elif dataset_name == self.DATASET_V2:
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_v2(rx_name, allowed_epochs)
        elif dataset_name == self.DATASET_V3:
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_v3(rx_name, allowed_epochs)
        elif dataset_name == self.DATASET_V4:
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_v4(rx_name, allowed_epochs)    
        elif dataset_name == self.DATASET_WISIG_OLD:
            # Note: in this case, dataset_train_path actually returns a list of two paths (day #1 and day #2)
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_wisig_old(equalized=wisig_equalized)
        elif dataset_name == self.DATASET_WISIG_NEW:
            # Note, in this case, we return a list of paths to h5 files without train/test split; the splitting will need to be done separately
            dataset_paths, model_path, samp_rate = self._load_dataset_wisig_new(equalized=wisig_equalized)
        elif dataset_name == self.DATASET_V2V4:
            # Note: in this case, dataset_train_path actually returns a list of two paths (day #1 and day #2)
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_v2v4(rx_name)
        else: print('Invalid dataset name.')

        if dataset_name == self.DATASET_WISIG_OLD:
            node_ids_train = [9, 11, 15, 17, 25, 38, 52, 57, 60, 69, 80, 84, 129, 130, 133, 142, 147, 157, 190, 196, 203, 206, 239, 242, 280, 300, 315, 329, 330, 360, 378, 380, 391]
            node_ids_epoch = [114, 159, 269, 266, 394, 398] if wisig_disjoint else [9, 11, 15, 17, 25, 38]
        elif dataset_name == self.DATASET_WISIG_NEW:
            # Note: all of these nodes are available on ALL 4 days of data capture
            node_ids_train = [60, 69, 80, 114, 130, 133, 142, 147, 196, 203, 206, 239, 266, 269, 280, 315, 380, 391, 398]
            node_ids_epoch = [9, 11, 15, 17, 38, 57] if wisig_disjoint else [269, 280, 315, 380, 391, 398]
            # node_ids_epoch = [269, 280, 315, 380, 391, 398] if wisig_disjoint else node_ids_train
            return dataset_paths, model_path, node_ids_train, node_ids_epoch, samp_rate
        elif dataset_name == self.DATASET_V2V4:
            # node_ids_train = self._get_dataset_devices(self.load_raw_dataset(dataset_train_path[0])[1], show=False)
            node_ids_train = [11, 17, 25, 38, 52, 57, 60, 129, 133, 147, 157, 159, 190, 196, 206, 216, 266, 391, 399]
            node_ids_epoch = [269, 398, 280, 315, 394, 300, 330]
        else:
            node_ids_train = self._get_dataset_devices(self.load_raw_dataset(dataset_train_path)[1], show=False)
            node_ids_epoch = self._get_dataset_devices(self.load_raw_dataset(dataset_epoch_paths[0])[1], show=False)

        # print(f"Devices for training: {len(node_ids_train)}")
        # print(f"Devices for testing: {len(node_ids_epoch)}")

        return dataset_train_path, dataset_epoch_paths, model_path, node_ids_train, node_ids_epoch, samp_rate

    def _shuffle_dataset(self, data, labels, rssi):
        # Produce a new order for elements
        new_order = np.arange(labels.shape[0])
        self.rng.shuffle(new_order)

        data = data[new_order, :]
        labels = labels[new_order]
        rssi = rssi[new_order] if rssi is not None else None

        return data, labels, rssi

    def load_raw_dataset(self, path, shuffle=False):
        with h5py.File(path,'r') as f:
            data = f['data'][:]
            rssi = f['rssi'][:] if 'rssi' in f else None
            labels = f['label'][:].astype(int)

            iq = data[:, 0::2] + 1j * data[:, 1::2] # Convert from interleaved doubles to complex values

        if shuffle:
            iq, labels, rssi = self._shuffle_dataset(iq, labels, rssi)
            
        return iq, labels, rssi

    def load_raw_dataset_wisig(self, path_train, path_test, shuffle=False):
        print(path_train)        
        print(path_test)

        iq_train, labels_train, rssi_train = self.load_raw_dataset(path_train, shuffle)
        iq_test, labels_test, rssi_test = self.load_raw_dataset(path_test, shuffle)

        iq = np.concatenate((iq_train, iq_test), axis=0)
        labels = np.concatenate((labels_train, labels_test), axis=0)
        rssi = np.concatenate((rssi_train, rssi_test), axis=0) if rssi_train is not None and rssi_test is not None else None

        return iq, labels, rssi

    def augment_dataset(self, data, labels, rssi, augment_cfo=False, multiplier=1):
        if augment_cfo:
            print(f'Augmenting the dataset: x{multiplier}, CFO range: [{ppm_range[0]}, {ppm_range[1]}], dist: {distribution}')
        else: print(f'Augmenting dataset: x{multiplier}, randomized CFO adding disabled.')

        # 1. Replicate the data K times
        data = np.repeat(data, multiplier, axis=0)
        labels = np.repeat(labels, multiplier, axis=0)
        if rssi is not None: rssi = np.repeat(rssi, multiplier, axis=0)

        # Generate CFO values for augmentation
        if augment_cfo:
            ppm_range = [-40, 40] # idealistic possible values
            # ppm_range = [-14, -2.5] # observed values
            distribution = 'uniform'
            freq = 2.4e9
            rnd = np.random.default_rng(42)

            cfo_values = cfo_utils.generate_cfo_values(
                n_samples=data.shape[0], 
                distribution=distribution,
                ppm_range=ppm_range,
                freq=freq,
                rnd=rnd,
                show=False)

            # 3. Apply generated CFO values to the dataset
            data = cfo_utils.compensate_cfo(data, cfo_values)

        return data, labels, rssi

    # def load_raw_dataset_wisig(self, path, shuffle=False, compensate_cfo=False, augment_cfo=False):
    #     iq, labels, rssi = self.load_raw_dataset(path, shuffle)
    #     iq = iq[:, 0:400] # keep only preambles

    #     # First (if needed), compensate CFO, before (optionally) applying randomly-generated CFO with a specified distribution
    #     if compensate_cfo:
    #         if 'non_eq' in path: # we're dealing with non-eq signal and simply need to remove CFO
    #             print('Removing CFO from raw signal.')
    #             cfo = cfo_utils.extract_data_cfo(iq)
    #             iq = cfo_utils.compensate_cfo(iq, cfo[:, 0] + cfo[:, 1])
    #         else: # we're dealing with equalized signal and need to extract CFO from non_eq
    #             print('Removing CFO from equalized signal.')
    #             iq_raw, _, _ = self.load_raw_dataset(path.replace('eq_', 'non_eq_'))
    #             cfo = cfo_utils.extract_data_cfo(iq_raw)
    #             iq = cfo_utils.compensate_cfo(iq, cfo[:, 0] + cfo[:, 1])
    #             # TODO: here we plot CFO before and after compensation across retrieved samples
    #             # cfo_new = cfo_utils.extract_data_cfo(iq)
    #             # plt.figure(figsize=(10, 8), dpi=80)
    #             # plt.plot(cfo[:, 0] + cfo[:, 1], 'red', label='Original CFO values')
    #             # plt.plot(cfo_new[:, 0] + cfo_new[:, 1], 'blue', label='Corrected CFO values (equalized!)')
    #             # plt.ylim(-50e3, 50e3)
    #             # plt.title('Day 2: CFO removal')
    #             # plt.show()
    #     else: print('Not removing CFO.')

    #     # Second (if needed), apply randomly-generated CFO -- but with a distribution that we control
    #     # multiplier = 1
    #     multiplier = 4
    #     iq, labels, rssi = self.augment_dataset(iq, labels, rssi, augment_cfo=augment_cfo, multiplier=multiplier)

    #     return iq, labels, rssi

    def filter_frames_by_rssi(self, data, labels, rssi, device_frames, show_dist=False):
        device_count = len(set(labels.flatten()))

        # Note: this function assumes that data is not shuffled and each device has equal # of frames
        rssi_idx_filtered_all = []
        for device_start_i in range(0, device_count * device_frames, device_frames):
            rssi_values = rssi[device_start_i : device_start_i + device_frames]
            rssi_idx_filtered_all.extend(device_start_i + utils.filter_abnormal_rssi(rssi_values, plot=show_dist))

        data = data[rssi_idx_filtered_all, :]
        labels = labels[rssi_idx_filtered_all]
        rssi = np.array(rssi[rssi_idx_filtered_all])

        print(f"Removed {device_count * device_frames - len(labels.flatten())} values.")

        return data, labels, rssi, rssi_idx_filtered_all

    def filter_frames_by_cfo(self, data, labels, rssi, show=False):
        data_filtered = np.zeros((0, data.shape[1]), dtype=np.complex128)
        labels_filtered = np.zeros((0, 1))
        if rssi is not None: rssi_filtered = np.zeros((0, 1))

        for device_id in list(set(labels.flatten())):
            device_idx = np.where(labels == device_id)[0]
            device_data = data[device_idx, :]
            device_labels = labels[device_idx]
            if rssi is not None: 
                device_rssi = rssi[device_idx]

            device_cfo = cfo_utils.extract_data_cfo(device_data)
            cfo_coarse_idx_filtered = utils.filter_abnormal_cfo(device_cfo[:, 0].squeeze(), plot=False)
            cfo_fine_idx_filtered = utils.filter_abnormal_cfo(device_cfo[:, 1].squeeze(), plot=False)
            cfo_idx_filtered = np.intersect1d(cfo_coarse_idx_filtered, cfo_fine_idx_filtered)

            if show:
                print(f"Device ID {device_id}: discarded {device_data.shape[0] - len(cfo_idx_filtered)} frames.")

            data_filtered = np.vstack((data_filtered, device_data[cfo_idx_filtered, :]))
            labels_filtered = np.vstack((labels_filtered, device_labels[cfo_idx_filtered]))
            if rssi is not None: 
                rssi_filtered = np.vstack((rssi_filtered, device_rssi[cfo_idx_filtered]))

        return data_filtered, labels_filtered, rssi_filtered if rssi is not None else None


    def load_augmented_dataset(self, path, samp_rate, aug_config, shuffle=False):
        # Note: before loading, ensure that there's a MATLAB session named 'mobintel_aug' (or update the name).
        #       To launch a session, run the following:
        #       - source ~/.bash_profile (or similar)
        #       - matlab -nodesktop -r "matlab.engine.shareEngine('mobintel_aug')"
        #       The system will either share a session with this name, or will say that such name is already used
        #       and instead will suggest an alternative name.
        result = self.mateng.augmentation(path, samp_rate, aug_config['t_rms_bounds'], aug_config['d_f_bounds'], aug_config['k_factor_bounds'], aug_config['multiplier'])
        
        data = np.array(result['data_aug']).swapaxes(0, 1)
        labels = np.array(result['label_aug']).swapaxes(0, 1)
        # rssi = np.array(result['rssi_aug']).swapaxes(0, 1)

        return data, labels, None

    def filter_dataset(self, data, labels, rssi, dev_range, pkt_range):
        # If the list of devices isn't specified - loop through all of the available ones
        if dev_range is None:
            dev_range = set(labels.flatten())

        try:
            # Filter indexes of frames to keep based on dev_range
            frame_idx_filtered = []
            for dev_idx in dev_range:
                # Find indexes of frames from a specified device
                frame_idx_device = np.where(labels.flatten()==int(dev_idx))[0]
                # Keep a specified sub-range of frames
                frame_idx_device = frame_idx_device[pkt_range]
                # Add these label indexes to our combined array
                frame_idx_filtered.extend(frame_idx_device)
            
            # Filter the dataset based on dev_range and pkt_range
            labels = labels[frame_idx_filtered]
            rssi = rssi[frame_idx_filtered] if rssi is not None else None
            data = data[frame_idx_filtered, :]

            return data, labels, rssi
        except Exception as e:
            return None, None, None

    def rssi_to_weight(self, rssi_dbm):
        # Convert RSSI to weighting factor by normalizing between [0, 1]
        # Note: weak signal is below 90 dBm, very strong signal is above -10 dBm
        if rssi_dbm < -100: 
            print(f'RSSI: {rssi_dbm}. Adjust normalization!')

        rssi_bounds = [-100, 0]
        rssi_scaled = (rssi_dbm - rssi_bounds[0]) / (rssi_bounds[1] - rssi_bounds[0])

        return rssi_scaled

    def load_testing_input(self, dataset_epoch_paths, epoch_idx, device_idx, frame_count):
        # Inputs:
        # - epoch_id:    1-based index of an epoch in the dataset
        # - device_id:   string name of the node (i.e., node10-5)
        # - frame_count: how many consecutive frames to return
        # Returns: a list of dict objects that have the following properties:
        # - iq:          complex-valued sample values for a given frame
        # - rssi:        RSSI value for the particular frame in dB

        # 1. Load data from a given epoch
        data, labels, rssi = self.load_raw_dataset(dataset_epoch_paths[epoch_idx], shuffle=False)

        # 2. Filter the epoch and only keep the device we need
        data, labels, rssi = self.filter_dataset(data, labels, rssi, dev_range=[device_idx], pkt_range=np.arange(frame_count))

        # 3. Reformat the dataset to a frame-based format
        if rssi is not None:
            return [{'iq': iq, 'rssi': rssi_value} for iq, rssi_value in zip(data, rssi)]
        else:
            return [{'iq': iq} for iq in data]

if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")