import os
import numpy as np
import queue
import concurrent.futures
import re
import h5py
import json
from collections import defaultdict
import matlab.engine
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = '/home/smazokha2016/Desktop'
WISIG_RAW_DIR = ROOT_DIR + '/wisig_raw'
RFFI_DATASET_TARGET_DIR = ROOT_DIR + '/wisig_dataset_new'
MATLAB_OFDM_DECODER_DIR = ROOT_DIR + '/mobintel-rffi/preprocessor/frame_mac_detection'
NODE_MACS_PATH = WISIG_RAW_DIR + '/experiment_device_macs.json'
FRAME_COUNT = 500

COMPLETED_SESSIONS = [
    # 'wifi_2021_03_01', 
    # 'wifi_2021_03_08', 
    # 'wifi_2021_03_15', 
    # 'wifi_2021_03_23'
]

# MATLAB_SESSION_NAMES = [
#     'mobintel_session_1',
#     'mobintel_session_2', 
#     'mobintel_session_3', 
#     'mobintel_session_4',
#     'mobintel_session_5',
#     'mobintel_session_6',
#     'mobintel_session_7',
#     'mobintel_session_8',
#     'mobintel_session_9',
#     'mobintel_session_10',
#     'mobintel_session_11',
#     'mobintel_session_12',
#     'mobintel_session_13',
#     'mobintel_session_14',
#     'mobintel_session_15']

MATLAB_SESSION_NAMES = [
    'MATLAB_141903',
    'MATLAB_145113',
    'MATLAB_147960',
    'MATLAB_148555',
    'MATLAB_149274',
    'MATLAB_150131',
    'MATLAB_150694',
    'MATLAB_151407',
    'MATLAB_152126',
    'MATLAB_153276'
]

MAX_THREADS = len(MATLAB_SESSION_NAMES)

# Extracts signal configs from a file name in a dataset
# - filename: name of the .dat file
def parse_dat_name(filename):
    filename = os.path.basename(filename)

    # Extract node_tx
    node_tx_match = re.search(r'tx\{node:(.*?)\}', filename)
    node_tx = node_tx_match.group(1) if node_tx_match else None

    # Extract node_rx
    node_rx_match = re.search(r'rx\{node:(.*?)-rxFreq', filename)
    node_rx = node_rx_match.group(1) if node_rx_match else None

    # Extract samp_rate
    samp_rate_match = re.search(r'rxSampRate:(\d+e\d+)', filename)
    samp_rate = float(samp_rate_match.group(1)) if samp_rate_match else None

    return {
        "node_tx": node_tx,
        "node_rx": node_rx,
        "samp_rate": samp_rate
    }

# Reads a JSON file containing MAC addresses of devices
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data

# Create a dictionary which contains IDs (1--400) and (X-Y) of all sensors physically present
# in the Orbit testbed facility. This is later used to produce unique labels for the sensor 
# fingerprinting model.
def generate_node_ids():
    ids = {}
    node_i = 0
    for i in np.arange(1, 21):
        for j in np.arange(1, 21):
            ids[str(i) + "-" + str(j)] = node_i
            node_i = node_i + 1
    return ids

# Save an h5 dataset file containing labels & data for a given set of devices
def save_dataset_h5(file_target, label, data, rssi):
    print('Saving', file_target)
    with h5py.File(file_target, 'w') as h5file:
        h5file.create_dataset('label', data=label, dtype='float64')
        h5file.create_dataset('rssi', data=rssi, dtype='float64')
        h5file.create_dataset('data', data=data, dtype='float64')  

# Package & store epoch infromation in h5 file (ready for RFFI)
def epoch_save(node_ids_dict, target_dir, epoch_preambles, session_name, preamble_len, enable_equalization):
    for rx_name in epoch_preambles.keys():
        rx_epochs = epoch_preambles[rx_name]

        # Data shape: (epochs x frames, samples x 2)
        # All frames/samples from all emitters are stitched together
        h5_data = np.zeros((len(rx_epochs) * FRAME_COUNT, preamble_len * 2), dtype='float64')
        # Labels shape: (epochs x frames, 1)
        h5_labels = np.zeros((len(rx_epochs) * FRAME_COUNT, 1), dtype='float64')
        # RSSI shape: (epochs x frames, 1)
        h5_rssi = np.zeros((len(rx_epochs) * FRAME_COUNT, 1), dtype='float64')

        h5_idx = 0
        for rx_epoch in rx_epochs:
            preambles = rx_epoch['preambles']
            rssi = rx_epoch['rssi']
            tx_node_name = rx_epoch['node_tx']
            for preamble_i in np.arange(0, preambles.shape[0]):
                h5_data[h5_idx, 0::2] = np.real(preambles[preamble_i, :])
                h5_data[h5_idx, 1::2] = np.imag(preambles[preamble_i, :])

                h5_labels[h5_idx] = node_ids_dict[tx_node_name]
                h5_rssi[h5_idx] = rssi[preamble_i]

                h5_idx = h5_idx + 1

        eq_chunk = 'eq' if enable_equalization else 'non_eq'
        dataset_filepath = os.path.join(target_dir, f'node{rx_name}_{eq_chunk}_{session_name}.h5')
        save_dataset_h5(dataset_filepath, h5_labels, h5_data, h5_rssi)

def process_dat_file(matlab_engine, dat_file, node_macs, preamble_len, enable_equalization):
    print(f"Processing {dat_file}")

    # 3.1. Extract signal info from its name
    dat_config = parse_dat_name(dat_file)
    tx_name = dat_config['node_tx'][4:]
    rx_name = dat_config['node_rx'][4:]
    samp_rate = dat_config['samp_rate']

    # # 3.2. Retrieve node MAC address
    # tx_mac = node_macs[tx_name]['mac']

    # tx_mac = '' # find the frames with most commonly found MAC address; yields mixed MACs
    tx_mac = 'e0:06:e6:18:45:cf' # 500 frames yields ~44 devices; 1000 frames yields ~16 devices
    # tx_mac = '00:15:6d:84:fe:9f' # 500 devices yields ~6 devices; 1000 frames yields ~3 devices

    # 3.3. Decode the file via MATLAB script, extract preambles
    response = matlab_engine.find_tx_frames(dat_file, 'CBW20', samp_rate, tx_mac, preamble_len, enable_equalization)
    # preamble_bounds = np.array(response['preamble_bounds']).squeeze()
    preamble_iq = np.array(response['preamble_iq']).squeeze()
    rssi = np.array(response['rssi']).squeeze()
    macs = np.array(response['macs']).squeeze()

    # If there's more than one MAC address found, let's filter them
    if len(set(macs)) > 1:
        print("More than one MAC address identified.")
        mac_unique, mac_counts = np.unique(macs, return_counts=True)
        for value, count in zip(mac_unique, mac_counts):
            print(f"{value}: {count}")
        
        # Pick the most commonly available MAC
        mac_choice = mac_unique[np.argmax(mac_counts)]
        print(f'Selecting {mac_choice}.')

        # Find item indexes with this MAC address
        mac_choice_idx = np.where(macs == mac_choice)[0]

        # Filter the data
        preamble_iq = preamble_iq[mac_choice_idx, :]
        rssi = rssi[mac_choice_idx]
        macs = macs[mac_choice_idx]

    # 3.4. Prepare information from a current dat file
    if preamble_iq.shape[0] >= FRAME_COUNT:
        file_preambles = {
            'preambles': preamble_iq[0:FRAME_COUNT, :],
            'rssi': rssi,
            'node_tx': tx_name,
            'node_rx': rx_name,
            'node_mac': macs[0]
        }
    else: file_preambles = None

    return file_preambles

def process_session(matlab_engine_queue, capture_session_path, preamble_len, node_ids, node_macs, enable_equalization):
    session_name = os.path.basename(capture_session_path)
    print(f'\n\n################ PROCESSING SESSION {session_name} ################\n\n')

    # Retrieve list of all .dat files to process
    session_dat_files = get_dat_files(capture_session_path)

    # Prepare a dictionary to store preambles for this session (aka epoch)
    epoch_preambles = defaultdict(list)

    # Define a worker function that would prepare a matlab engine for work
    def worker(dat_file, node_macs, preamble_len):
        # Retrieve name of the session
        matlab_engine = matlab_engine_queue.get()
        dat_file_preambles = None
        try:
            dat_file_preambles = process_dat_file(matlab_engine, dat_file, node_macs, preamble_len, enable_equalization)
        except Exception as e:
            print(f"Something happened: {dat_file}")
            print(e)
        finally:
            matlab_engine_queue.put(matlab_engine)
        return (dat_file_preambles, dat_file)

    # Initialize parallel analysis for all .dat files
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(worker, dat_file, node_macs, preamble_len) for dat_file in session_dat_files]
        # TODO DEBUG: futures = [executor.submit(worker, dat_file, node_macs, preamble_len) for dat_file in [session_dat_files[0]]]
        concurrent.futures.wait(futures)

        for future in concurrent.futures.as_completed(futures):
            dat_file_preambles, dat_file = future.result()

            if dat_file_preambles:
                rx_name = dat_file_preambles['node_rx']
                epoch_preambles[rx_name].append(dat_file_preambles)
            else: print(f"Insufficient frames captured: {dat_file}")

    # Save information for this session/epoch into a final dataset file
    epoch_save(node_ids, RFFI_DATASET_TARGET_DIR, epoch_preambles, session_name, preamble_len, enable_equalization)

    print(f"Session {capture_session_path} processing is complete.")
    print("=========================================================================")

def get_dat_files(directory):
    """
    Recursively find all .dat files in the given directory and its subdirectories.
    Returns a list of relative paths to the .dat files.
    """
    dat_files = []

    # Check if a directory to store final dataset exists and create if not
    if not os.path.exists(directory):
        print(f'Such directory doesn\'t exist: {directory}')
        return []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('tx{') and file.endswith('.dat'):
                dat_files.append(os.path.join(directory, root, file))
    
    return sorted(dat_files) 

def get_directories(directory):
    return sorted([subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))])

def request_preamble_len():
    try:
        return int(input("What should be preamble length? [400] "))
    except:
        return 400

def main():
    # preamble_len = request_preamble_len()
    preamble_len = 400
    enable_equalization = True

    # Check if a directory to store final dataset exists and create if not
    if not os.path.exists(RFFI_DATASET_TARGET_DIR):
        print(f'Creating destination directory: {RFFI_DATASET_TARGET_DIR}')
        os.makedirs(RFFI_DATASET_TARGET_DIR)

    # Generate a dictionary of node IDs
    node_ids = generate_node_ids()

    # Obtain a list of capture files to process
    capture_sessions = get_directories(WISIG_RAW_DIR)

    # Load a JSON file with device MAC addresses from S3 experiment folder
    node_macs = read_json_file(NODE_MACS_PATH)

    # Initialize a queue that would store all available Matlab engine instances
    matlab_engine_queue = queue.Queue()
    for engine_name in MATLAB_SESSION_NAMES:
        print(f"Connecting to engine {engine_name}... ", end='')
        matlab_engine = matlab.engine.connect_matlab(engine_name)
        # matlab_engine = matlab.engine.start_matlab("-nodisplay")
        matlab_engine.cd(MATLAB_OFDM_DECODER_DIR, nargout=0)
        matlab_engine_queue.put(matlab_engine)
        print("connected")

    # Work throughe each session (aka day of data)
    for capture_session in capture_sessions:
        if capture_session in COMPLETED_SESSIONS:
            print(f"Capture session {capture_session} already completed.")
            continue
        else: 
            capture_session_path = os.path.join(WISIG_RAW_DIR, capture_session)
            process_session(matlab_engine_queue, capture_session_path, preamble_len, node_ids, node_macs, enable_equalization)

if __name__ == "__main__":
    main()