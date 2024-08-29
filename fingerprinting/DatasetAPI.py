import os
import numpy as np
import h5py
import matlab.engine
from Singleton import Singleton

class DatasetAPI(metaclass=Singleton):

    DATASET_V1 = 'v1_jul_13' # 25 Msps, 2-hour period, manual capture, only one RX node (node1-1)
    DATASET_V2 = 'v2_jul_19' # 25 Msps, 24-hour period, automated capture, 4 RX nodes
    DATASET_V3 = 'v2_jul_21' # 20 Msps, 4-hour period, automated capture, 4 RX nodes
    DATASET_V4 = 'v3_aug_8'  # 25 Msps, 24-hour period, automated capture, 3 RX nodes (1-1, 1-20, 19-19)

    RX_1 = 'node1-1'
    RX_2 = 'node1-20'
    RX_3 = 'node20-1'
    RX_4 = 'node19-19'

    def __init__(self, root_dir, matlab_src_dir, matlab_session_id, aug_on=False):
        self.root_dir = root_dir
        self.dataset_v1_path = os.path.join(self.root_dir, 'orbit_dataset_v1')
        self.dataset_v2_path = os.path.join(self.root_dir, 'orbit_dataset_v2_jul19_full')
        self.dataset_v3_path = os.path.join(self.root_dir, 'orbit_dataset_v2_jul21')
        self.dataset_v4_path = os.path.join(self.root_dir, 'mobintel-orbit-dataset_h5')

        if aug_on:
            mateng = matlab.engine.connect_matlab(matlab_session_id)
            mateng.cd(matlab_src_dir, nargout=0)
        else: self.mateng = None

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

    def load_dataset_info(self, dataset_name, rx_name, allowed_epochs):
        if dataset_name == self.DATASET_V1:
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_v1(rx_name, allowed_epochs)
        elif dataset_name == self.DATASET_V2:
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_v2(rx_name, allowed_epochs)
        elif dataset_name == self.DATASET_V3:
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_v3(rx_name, allowed_epochs)
        elif dataset_name == self.DATASET_V4:
            dataset_train_path, dataset_epoch_paths, model_path, samp_rate = self._load_dataset_v4(rx_name, allowed_epochs)    
        else: print('Invalid dataset name.')

        node_ids_train = self._get_dataset_devices(self.load_raw_dataset(dataset_train_path)[1], show=False)
        node_ids_epoch = self._get_dataset_devices(self.load_raw_dataset(dataset_epoch_paths[0])[1], show=False)

        return dataset_train_path, dataset_epoch_paths, model_path, node_ids_train, node_ids_epoch, samp_rate

    def _shuffle_dataset(self, data, labels, rssi):
        # Produce a new order for elements
        new_order = np.arange(labels.shape[0])
        np.random.shuffle(new_order)

        data = data[new_order, :]
        labels = labels[new_order]
        rssi = rssi[new_order]

        return data, labels, rssi

    def load_raw_dataset(self, path, shuffle=False):
        with h5py.File(path,'r') as f:
            data = f['data'][:]
            rssi = f['rssi'][:]
            labels = f['label'][:].astype(int)

            iq = data[:, 0::2] + 1j * data[:, 1::2] # Convert from interleaved doubles to complex values

        if shuffle:
            iq, labels, rssi = self._shuffle_dataset(iq, labels, rssi)
            
        return iq, labels, rssi

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
        rssi = np.array(result['rssi_aug']).swapaxes(0, 1)

        return data, labels, rssi

    def filter_dataset(self, data, labels, rssi, dev_range, pkt_range):
        # If the list of devices isn't specified - loop through all of the available ones
        if dev_range is None:
            dev_range = set(labels.flatten())

        # Filter indexes of frames to keep based on dev_range
        frame_idx_filtered = []
        for dev_idx in dev_range:
            # Extract only the specified devices, and for each only pkt_range frames
            frame_idx_device = np.where(labels==int(dev_idx))[0][pkt_range]
            frame_idx_filtered.extend(frame_idx_device)
        
        # Filter the dataset based on dev_range and pkt_range
        labels = labels[frame_idx_filtered]
        rssi = rssi[frame_idx_filtered]
        data = data[frame_idx_filtered, :]

        return data, labels, rssi

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
        return [{'iq': iq, 'rssi': rssi_value} for iq, rssi_value in zip(data, rssi)]

if __name__ == "__main__":
    print("Please refer to the primary workbook or the README for tutorial on how to use this class.")