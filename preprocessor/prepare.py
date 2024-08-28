import os
import numpy as np
import queue
import concurrent.futures
import re
import h5py
import json
from collections import defaultdict
import boto3
from tqdm import tqdm
import matlab.engine
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client('s3')

ROOT_DIR = '/home/smazokha2016/Desktop'
# ROOT_DIR = '/Users/stepanmazokha/Desktop'

MATLAB_OFDM_DECODER = ROOT_DIR + '/mobintel-rffi/preprocessor/frame_mac_detection'
TEMP_IQ_DIRECTORY = ROOT_DIR + '/orbit_processor_temp/'
NODE_MACS = 'experiment_device_macs.json'
S3_BUCKET_NAME = "mobintel-orbit-dataset"
S3_EXPERIMENT_NAME = "orbit_experiment_aug_8"
# S3_EXPERIMENT_NAME = "orbit_experiment_jul_19"
# S3_EXPERIMENT_NAME = "orbit_experiment_jul_21"
S3_EPOCH_PREFIX = "epoch_"
S3_TRAINING_PREFIX = "training_"
RFFI_DATASET_TARGET_DIR = f'{ROOT_DIR}/{S3_BUCKET_NAME}_h5/'
FRAME_COUNT = 100

COMPLETED_SESSIONS = [
    'training_2024-08-08_18-37-33',
    'epoch_2024-08-08_19-19-27',
    'epoch_2024-08-08_19-59-37',
    'epoch_2024-08-08_20-33-18',
    'epoch_2024-08-08_21-04-40',
    'epoch_2024-08-08_21-40-14',
    'epoch_2024-08-08_22-15-27',
    'epoch_2024-08-08_22-45-31',
    'epoch_2024-08-08_23-15-19',
    'epoch_2024-08-08_23-43-47',
    'epoch_2024-08-09_00-15-50',
    'epoch_2024-08-09_00-44-03',
    'epoch_2024-08-09_01-12-19',
    'epoch_2024-08-09_01-40-27',
    'epoch_2024-08-09_02-08-14',
    'epoch_2024-08-09_02-36-31',
    'epoch_2024-08-09_03-05-59',
    'epoch_2024-08-09_03-34-38',
    'epoch_2024-08-09_04-02-08',
    'epoch_2024-08-09_04-30-31',
    'epoch_2024-08-09_04-59-00',
    'epoch_2024-08-09_05-27-49',
    'epoch_2024-08-09_05-56-24',
    'epoch_2024-08-09_06-24-29',
    'epoch_2024-08-09_06-52-36',
    'epoch_2024-08-09_07-20-51',
    'epoch_2024-08-09_07-49-48',
    'epoch_2024-08-09_08-18-31',
    'epoch_2024-08-09_08-47-08',
    'epoch_2024-08-09_09-15-25',
    'epoch_2024-08-09_14-11-49'
]

MATLAB_SESSION_NAMES = [
    'mobintel_session_1',
    'mobintel_session_2', 
    'mobintel_session_3', 
    'mobintel_session_4',
    'mobintel_session_5',
    'mobintel_session_6',
    'mobintel_session_7',
    'mobintel_session_8',
    'mobintel_session_9',
    'mobintel_session_10']

MAX_THREADS = len(MATLAB_SESSION_NAMES)

# Extracts signal configs from a file name in a dataset
# - filename: name of the .dat file (without the route)
def parse_dat_name(filename):
    # Extract node_tx
    node_tx_match = re.search(r'tx\{node_(.*?)\}', filename)
    node_tx = node_tx_match.group(1) if node_tx_match else None

    # Extract node_rx
    node_rx_match = re.search(r'rx\{node_(.*?)[\+\}]', filename)
    node_rx = node_rx_match.group(1) if node_rx_match else None

    # Extract samp_rate
    samp_rate_match = re.search(r'rxSampRate_(\d+e\d+)', filename)
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

class TqdmCallback:
    def __init__(self, total_size):
        self.progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading', bar_format='{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}/{remaining}]', ascii=' █')

    def __call__(self, bytes_amount):
        self.progress_bar.update(bytes_amount)

def download_file_with_progress(bucket_name, s3_key, local_path):
    s3 = boto3.client('s3')

    # Get the total size of the object
    response = s3.head_object(Bucket=bucket_name, Key=s3_key)
    total_size = response['ContentLength']

    # Ensure the local directory exists
    local_dir = os.path.dirname(local_path)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Start the download and show the progress
    callback = TqdmCallback(total_size)
    s3.download_file(bucket_name, s3_key, local_path, Callback=callback)
    callback.progress_bar.close()

def s3_list_subdirs(bucket_name, prefix):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    
    subdirs = []
    for path in response['CommonPrefixes']:
        subdirs.append(os.path.basename(os.path.normpath(path['Prefix'])))
    return subdirs

def s3_list_files(bucket_name, prefix):
    # Initialize the paginator
    paginator = s3.get_paginator('list_objects_v2')

    # Create a PageIterator from the Paginator
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    # List to store all file keys
    filenames = []

    # Iterate through each page
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                filename = os.path.basename(os.path.normpath(obj['Key']))
                if filename[-4:] == '.dat':
                    filenames.append(filename)

    return filenames

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
def epoch_save(node_ids_dict, target_dir, epoch_preambles, session_name, preamble_len):
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

        dataset_filepath = os.path.join(target_dir, f'node{rx_name}_{session_name}.h5')
        save_dataset_h5(dataset_filepath, h5_labels, h5_data, h5_rssi)

def is_session_valid(session_name):
    return session_name[0:6] == 'epoch_' or session_name[0:9] == 'training_'
    
def process_dat_file(matlab_engine, session_name, dat_file, node_macs, preamble_len):
    print(f"Processing {dat_file}")

    # 3.1. Download the file from S3
    s3_filepath = f"{S3_EXPERIMENT_NAME}/{session_name}/{dat_file}"
    local_filepath = os.path.join(TEMP_IQ_DIRECTORY, dat_file)
    print(f'Downloading {dat_file}...')
    download_file_with_progress(S3_BUCKET_NAME, s3_filepath, local_filepath)

    # 3.2. Extract signal info from its name
    dat_config = parse_dat_name(dat_file)
    tx_name = dat_config['node_tx'][4:]
    rx_name = dat_config['node_rx'][4:]
    samp_rate = dat_config['samp_rate']

    # 3.3. Retrieve node MAC address
    tx_mac = node_macs[tx_name]['mac']

    # 3.2. Decode the file via MATLAB script, extract preambles
    response = matlab_engine.find_tx_frames(local_filepath, 'CBW20', samp_rate, tx_mac, preamble_len)
    # preamble_bounds = np.array(response['preamble_bounds']).squeeze()
    preamble_iq = np.array(response['preamble_iq']).squeeze()
    rssi = np.array(response['rssi']).squeeze()

    # 3.3. Prepare information from a current dat file
    if preamble_iq.shape[0] >= FRAME_COUNT:
        file_preambles = {
            'preambles': preamble_iq[0:FRAME_COUNT, :],
            'rssi': rssi,
            'node_tx': tx_name,
            'node_rx': rx_name,
            'node_mac': tx_mac
        }
    else: file_preambles = None

    # 3.4. Remove local file afer the processing is completed
    print(f"Deleting local file {local_filepath}")
    os.remove(local_filepath)

    return file_preambles

def process_session(matlab_engine_queue, session_name, preamble_len, node_ids, node_macs):
    if not is_session_valid(session_name):
        print("Skipping session", session_name)
        return
    else: print("Processing session ", session_name)

    # Retrieve list of all .dat files to process
    session_dat_files = s3_list_files(S3_BUCKET_NAME, S3_EXPERIMENT_NAME + "/" + session_name + "/")

    # Prepare a dictionary to store preambles for this session (aka epoch)
    epoch_preambles = defaultdict(list)

    # Define a worker function that would prepare a matlab engine for work
    def worker(session_name, dat_file, node_macs, preamble_len):
        # Retrieve name of the session
        matlab_engine = matlab_engine_queue.get()
        dat_file_preambles = None
        try:
            dat_file_preambles = process_dat_file(matlab_engine, session_name, dat_file, node_macs, preamble_len)
        except Exception as e:
            print(f"Something happened: {dat_file}")
            print(e)
        finally:
            matlab_engine_queue.put(matlab_engine)
        return (dat_file_preambles, dat_file)

    # Initialize parallel analysis for all .dat files
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(worker, session_name, dat_file, node_macs, preamble_len) for dat_file in session_dat_files]
        concurrent.futures.wait(futures)

        for future in concurrent.futures.as_completed(futures):
            dat_file_preambles, dat_file = future.result()

            if dat_file_preambles:
                rx_name = dat_file_preambles['node_rx']
                epoch_preambles[rx_name].append(dat_file_preambles)
            else: print(f"Insufficient frames captured: {dat_file}")

    # Save information for this session/epoch into a final dataset file
    epoch_save(node_ids, RFFI_DATASET_TARGET_DIR, epoch_preambles, session_name, preamble_len)

    print(f"Session {session_name} processing is complete.")
    print("=========================================================================")

def request_preamble_len():
    try:
        return int(input("What should be preamble length? [400] "))
    except:
        return 400
    
def request_mode_session():
    while True:
        mode = input("Which mode should we run? [single | full]")

        if mode == 'single':
            session_name = input("Which session should we process? [e.g., epoch_....]")
            if is_session_valid(session_name):
                return session_name
            else: print("Invalid session name.")
        elif mode == 'full':
            return None
        else: print('Invalid command.')

def main():
    preamble_len = request_preamble_len()

    # Check if a directory to store final dataset exists and create if not
    if not os.path.exists(RFFI_DATASET_TARGET_DIR):
        os.makedirs(RFFI_DATASET_TARGET_DIR)

    # Load a JSON file with device MAC addresses from S3 experiment folder
    node_macs_local_path = os.path.join(RFFI_DATASET_TARGET_DIR, NODE_MACS)
    download_file_with_progress(S3_BUCKET_NAME, f"{S3_EXPERIMENT_NAME}/{NODE_MACS}", node_macs_local_path)
    node_macs = read_json_file(node_macs_local_path)

    # Generate a dictionary of node IDs
    node_ids = generate_node_ids()

    # Let the user chose whether to run all sessions (from S3) or just one
    requested_session = request_mode_session()
    if requested_session: 
        sessions = [requested_session] 
    else: 
        sessions = s3_list_subdirs(S3_BUCKET_NAME, S3_EXPERIMENT_NAME + '/')
    print(f"Starting to process {len(sessions)} sessions.")

    # Initialize a queue that would store all available Matlab engine instances
    matlab_engine_queue = queue.Queue()
    for engine_name in MATLAB_SESSION_NAMES:
        print(f"Connecting to engine {engine_name}... ", end='')
        # matlab_engine = matlab.engine.connect_matlab(engine_name)
        matlab_engine = matlab.engine.start_matlab("-nodisplay")
        matlab_engine.cd(MATLAB_OFDM_DECODER, nargout=0)
        matlab_engine_queue.put(matlab_engine)
        print("connected")

    # Work throughe each session (aka training / testing epochs)
    for session_name in sessions:
        if session_name in COMPLETED_SESSIONS:
            print(f"Session {session_name} already completed.")
            continue
        else: 
            process_session(matlab_engine_queue, session_name, preamble_len, node_ids, node_macs)

if __name__ == "__main__":
    main()