# This is a master script. It performs a complete suite of tasks to configure nodes,
# transmit and receive samples, ans so forth. The objective of this script is to 
# perform the entire process of data capture completely automatically, without 
# any involvement of an operator.

import os
import tx_udp_master
import tx_probe_master
import rx_master
import threading
import random
import numpy as np
import queue
import time
from s3_uploader import S3Uploader
from concurrent.futures import ThreadPoolExecutor, as_completed

AP_NODE = "node2-5"
# RX_NODES = ["node1-1", "node1-20", "node20-1", "node19-19"]
RX_NODES = ["node1-1", "node1-20", "node19-19"]
# TX_TRAINING_NODES = ['node5-1', 'node7-10', 'node7-14', 'node2-19', 'node5-5', 'node19-1', 'node20-20', 'node1-10', 'node8-20', 'node11-17', 
#                      'node2-6', 'node1-12', 'node4-1', 'node3-13', 'node1-16', 'node8-8', 'node8-18', 'node1-19', 'node1-18', 'node11-7', 
#                      'node20-12', 'node4-10', 'node11-4', 'node8-3', 'node4-11', 'node3-18', 'node14-7', 'node10-17', 'node10-11']
# TX_TESTING_NODES = ['node12-20', 'node17-11', 'node20-19', 'node20-15', 'node14-10', 'node16-16', 'node15-1', 'node16-1', 'node1-2', 'node1-11', 'node2-1', 'node2-20'] 

TX_TRAINING_NODES = [
    'node7-10', 'node7-14', 'node2-19', 'node20-20', 'node8-20', 'node11-17', 'node2-6', 'node20-6', 'node15-20', 'node17-10',
    'node1-12', 'node4-1', 'node3-13', 'node1-16', 'node8-8', 'node8-18', 'node1-18', 'node11-7', 'node20-12', 'node19-2'
    'node11-4', 'node3-18', 'node14-7', 'node10-17', 'node10-11', 'node6-15', 'node7-7', 'node8-13', 'node10-1', 'node20-10']

# 'node19-1', 'node1-10'
# node7-7, node20-6, node20-10 -- these nodes are present in the dataset but don't have any data in them

TX_TESTING_NODES = [
    'node12-20', 'node17-11', 'node20-19', 'node20-15', 'node14-10', 'node16-16', 'node15-1', 'node16-1', 'node1-2', 'node1-11', 
    'node2-1', 'node2-20', 'node10-7', 'node10-10', 'node11-20', 'node11-10', 'node13-13', 'node11-14', 'node13-8', 'node13-18', 
    'node13-20', 'node14-11', 'node14-14', 'node19-20']

# Bad ones: node10-4, node12-1, node13-3, node15-15

TX_INTERVAL = "0.01" # Interval (in seconds) between injected probe requests
TX_SSID = "smazokha" # Name of the SSID which we'll use in the probe requests (irrelevant)
TX_MAC = "11:22:33:44:55:66" # Spoofed MAC address we'll use in our probe requests
TX_CHANNEL = 11 # Channel ID on which we'll be sending our probes [1 -- 13]
RX_CAP_LEN_UDP = "4" # For how many seconds should we capture UDP traffic
RX_CAP_LEN_PROBES = "10" # For how many seconds should we capture Probe Request traffoc
CONFIG_BATCH_SIZE = 15 # How many parallel config sessions should we run in parallel
AWS_S3_BUCKET_NAME = 'mobintel-orbit-dataset'
# EXPERIMENT_DIR = '/Users/stepanmazokha/Desktop/orbit_experiment/' # Root experiment dir on local device
EXPERIMENT_DIR = '/home/smazokha2016/Desktop/orbit_experiment_aug_8/' # Root experiment dir on CA-AI server

# Generates a 'virtual' MAC address (first 3 octets are the same, the remaining are randomized)
def generate_virtual_mac():
    random_octets = [random.randint(0x00, 0xFF) for _ in range(3)]
    random_mac_part = ':'.join(f'{octet:02x}' for octet in random_octets)
    return f'11:22:33:{random_mac_part}'

# Performs simultaneous signal capture on all specified RX devices
# - tx_node_id: identifier of the transmitting node (format: X-Y)
# - rx_node_id: identifier of the receiving node (format: X-Y)
# - target_dir: directory where the signal should be stored
# - start_event, stop_event: thread events to control task execution
def command_rx(tx_node_id, rx_node_id, cap_len_sec, target_dir, start_event):
    print(f"RX: {tx_node_id} -> {rx_node_id}: Starting...")
    try:
        # Wait for the start signal
        start_event.wait()

        print(f"RX: {tx_node_id} -> {rx_node_id}: Running...")

        rx_file = rx_master.node_capture(tx_node_id, rx_node_id, target_dir, cap_len_sec)

        print(f"RX: {tx_node_id} -> {rx_node_id}: Completed.")

        return rx_file
    except Exception as e:
        print(f"Something went wrong: {str(e)}")
        return None

# Performs configuration of a given device:
# - node_id: identifier of the configured node (format: X-Y)
# - node_type: type of the node [AP, TX-probe, TX-udp, RX]
def command_config(node_id, node_type, channel):
    print(f"Config: {node_id}, {node_type} started")
    try:
        if node_type == 'AP': # only relevant in case we're sending UDP traffic
            tx_udp_master.node_configure_ap(node_id, driver_name=tx_udp_master.WIFI_DRIVER_ATHEROS_10k, channel=channel)
        elif node_type == 'TX-probe':
            tx_probe_master.node_configure(node_id, driver_name=tx_probe_master.WIFI_DRIVER_ATHEROS_MAIN)
        elif node_type == 'TX-udp':
            tx_udp_master.node_configure_tx(node_id, driver_name=tx_udp_master.WIFI_DRIVER_ATHEROS_MAIN)
        elif node_type == 'RX':
            rx_master.node_configure(node_id)
        else: 
            print("Invalid node type.")

        print(f"Config: {node_id} ({node_type}) finished")
    except Exception as e:
        print(f"Something went wrong ({node_id}, {node_type}): {str(e)}")

# Runs configuration of all relevant nodes for the experiment
# - tx_node_ids: list of TX
# - rx_node_ids: list of RX nodes
# - ap_node_ids: list of AP nodes (either one entry or empty)
# - tx_mode: type of TX traffic to be sent [probe | udp]
def run_config(tx_node_ids, rx_node_ids, ap_node_ids, tx_mode, batch_size, tx_channel):
    config_queue = queue.Queue()
    [config_queue.put((node_id, 'RX')) for node_id in rx_node_ids]
    [config_queue.put((node_id, 'TX-' + tx_mode)) for node_id in tx_node_ids]
    [config_queue.put((node_id, 'AP')) for node_id in ap_node_ids]

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        while not config_queue.empty():
            if len(futures) < batch_size:
                node_id, node_type = config_queue.get()
                future = executor.submit(command_config, node_id, node_type, tx_channel)
                futures.append(future)
            else:
                # Check and remove completed futures
                for future in as_completed(futures):
                    futures.remove(future)
                    break  # Only remove one future per iteration to maintain the batch size

        # Ensure all remaining futures are completed
        for future in as_completed(futures):
            pass

# Runs signal capture across specified RX nodes
# - rx_node_ids: identifiers of the RX nodes that are capturing signal
# - tx_node_id: identifier of the TX node that emits signal (for file naming only)
# - target_dir: directory where we'll save the file with IQ samples
# - the function returns the list of file names which were produced in the target_dir
def run_rx(tx_node_id, rx_node_ids, cap_len_sec, target_dir):
    start_event = threading.Event()
    rx_files = []

    # 1. Configure and start all threads
    with ThreadPoolExecutor(max_workers=len(rx_node_ids)) as executor:
        futures = []
        for rx_node_id in rx_node_ids:
            future = executor.submit(command_rx, tx_node_id, rx_node_id, cap_len_sec, target_dir, start_event)
            futures.append(future)

        # 2. Launch signal capture on all threads simultaneously
        start_event.set()

        # 3. Collect the results as the threads complete
        for future in as_completed(futures):
            rx_files.append(future.result())

    return rx_files

# Runs a single transmission task, probe request mode, store on a local device
def run_capture_probes(tx_node_id, rx_node_ids, channel, ssid, interval, target_dir, cap_len_sec):
    mac = generate_virtual_mac() # produce a randomized (aka 'virtual') MAC address, unqiue for each epoch/device

    # Start transmission
    wifi_interface = tx_probe_master.node_emit_start(tx_node_id, channel, mac, ssid, interval)
    time.sleep(5) # wait while the tmux session starts on TX device

    # Perform capture
    rx_files = run_rx(tx_node_id, rx_node_ids, cap_len_sec, target_dir)

    # Stop transmission
    tx_probe_master.node_emit_stop(tx_node_id, wifi_interface)

    return rx_files

# Runs a single transmission task, udp mode, store on a local device
def run_capture_udp(tx_node_id, ap_node_id, rx_node_ids, target_dir, cap_len_sec):
    # Start transmission
    tx_udp_master.node_transmission_start(tx_node_id, ap_node_id)
    time.sleep(5) # wait while the tmux session starts on TX device

    # Perform capture
    rx_files = run_rx(tx_node_id, rx_node_ids, cap_len_sec, target_dir)

    # Stop transmission
    tx_udp_master.node_transmission_stop(tx_node_id, ap_node_id)

    return rx_files

# Uploads produced sample data to S3
# - bucket_name: name of the S3 bucket
# - experiment_dir: path to the local experiment folder
# - epoch_name: name of the RX session (training or testing epoch)
# - rx_files: list of full paths to .dat files to upload
#
# S3 path to each file looks like this:
#     {experiment_name}/{epoch_name}/{rx_file.dat}
#
#     Note: no '/' at the beginning!
#     Note: if there are files with the same "key" in AWS -- they will be overwritten
def upload_samples(bucket_name, experiment_dir, target_dir, rx_files):
    experiment_name = os.path.basename(experiment_dir.rstrip('/'))
    epoch_name = os.path.basename(target_dir.rstrip('/'))

    s3_file_paths = []
    for local_path in rx_files:
        local_filename = os.path.basename(local_path)
        s3_filename = f"{experiment_name}/{epoch_name}/{local_filename}"
        s3_file_paths.append(s3_filename)

    S3Uploader().upload_files_to_s3(bucket_name, rx_files, s3_file_paths)

# Deletes local samples
def delete_local_samples(rx_files):
    for rx_file in rx_files:
        try:
            if os.path.isfile(rx_file):
                os.remove(rx_file)
                print(f"Deleted {rx_file}")
        except Exception as e:
            print(f"Failed to delete {rx_file}. Reason: {e}")

# Runs a full experiment
# - tx_node: [probe | udp]
def run_full_experiment(tx_node_ids_train, tx_node_ids_test, rx_node_ids, ap_node_id, experiment_dir, epochs, tx_mode, cloud_sync=True, s3_bucket_name=AWS_S3_BUCKET_NAME):
    input("Ready to start capturing training data?")

    target_dir = rx_master.prepare_target_dir(experiment_dir, 'training_')
    os.makedirs(target_dir, exist_ok=True)

    # 1. Perform data capture for the training devices
    print(f"Running training capture")

    rx_files_all = []
    for tx_node_id in tx_node_ids_train:
        if tx_mode == 'udp':
            rx_files = run_capture_udp(tx_node_id, ap_node_id, rx_node_ids, target_dir, RX_CAP_LEN_UDP)
        else: 
            rx_files = run_capture_probes(tx_node_id, rx_node_ids, TX_CHANNEL, TX_SSID, TX_INTERVAL, target_dir, RX_CAP_LEN_PROBES)

        if len(rx_files) < len(rx_node_ids) or None in rx_files:
            print("Some files are missing! Check the directory before moving forward")
        rx_files_all.extend(rx_files)

    if cloud_sync:
        print("Uploading training data to S3...")
        upload_samples(s3_bucket_name, experiment_dir, target_dir, rx_files_all)
        delete_local_samples(rx_files_all)

    input("Ready to start capturing epochs?")
    print("================ TRAINING CAPTURE COMPLETE ================")

    # 2. Run testing data capture for a given number of epochs
    for epoch_i in np.arange(epochs):
        print(f"Running epoch #{epoch_i + 1}")

        rx_files_all = [] # resetting the list of RX file paths

        target_dir = rx_master.prepare_target_dir(experiment_dir, 'epoch_')
        os.makedirs(target_dir, exist_ok=True)
        for tx_node_id in tx_node_ids_test:
            if tx_mode == 'udp':
                rx_files = run_capture_udp(tx_node_id, ap_node_id, rx_node_ids, target_dir, RX_CAP_LEN_UDP)
            else: # "probe"
                rx_files = run_capture_probes(tx_node_id, rx_node_ids, TX_CHANNEL, TX_SSID, TX_INTERVAL, target_dir, RX_CAP_LEN_PROBES)
                
            if len(rx_files) < len(rx_node_ids) or None in rx_files:
                print("Some files are missing! Check the directory before moving forward")
            rx_files_all.extend(rx_files)

        if cloud_sync:
            print("Uploading epoch data to S3...")
            upload_samples(s3_bucket_name, experiment_dir, target_dir, rx_files_all)
            delete_local_samples(rx_files_all)
            
        print(f"================ EPOCH #{epoch_i + 1} CAPTURE COMPLETE ================")

def main():
    # Ensure that the dir for the experiment is correct
    input(f"Experiment dir: {EXPERIMENT_DIR}. OK?")
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    os.system(f"ssh -o StrictHostKeyChecking=no {tx_udp_master.JUMP_NODE_GRID} rm -rf ~/.ssh/known_hosts")
    os.system(f"ssh -o StrictHostKeyChecking=no {tx_udp_master.JUMP_NODE_OUTDOOR} rm -rf ~/.ssh/known_hosts")

    while True:
        instruction = input("What should we do? [config [probe | udp] | emit [probe | udp]] | run experiment [probe | udp]")

        if instruction == 'config probe':
            run_config(tx_node_ids=(TX_TRAINING_NODES + TX_TESTING_NODES), rx_node_ids=RX_NODES, ap_node_ids=[], tx_mode='probe', batch_size=CONFIG_BATCH_SIZE, tx_channel=TX_CHANNEL)
        elif instruction == 'config udp':
            run_config(tx_node_ids=(TX_TRAINING_NODES + TX_TESTING_NODES), rx_node_ids=RX_NODES, ap_node_ids=[AP_NODE], tx_mode='udp', batch_size=CONFIG_BATCH_SIZE, tx_channel=TX_CHANNEL)
        elif instruction == 'emit probe':
            tx_node_id = input('TX node ID: ') # 'node14-7'
            target_dir = rx_master.prepare_target_dir(EXPERIMENT_DIR, '')
            os.makedirs(target_dir, exist_ok=True)
            run_capture_probes(tx_node_id, RX_NODES, TX_CHANNEL, TX_MAC, TX_SSID, TX_SSID, target_dir)
        elif instruction == 'emit udp':
            tx_node_id = input('TX node ID: ') # 'node14-7'
            ap_node_id = input('AP node ID: ') # AP_NODE
            target_dir = rx_master.prepare_target_dir(EXPERIMENT_DIR, '')
            os.makedirs(target_dir, exist_ok=True)
            run_capture_udp(tx_node_id, ap_node_id, RX_NODES, target_dir, RX_CAP_LEN_UDP)
        elif instruction == 'run experiment probe':
            epochs = int(input('How many epochs? (int only please): '))
            cloud_sync = input('Should we record to AWS S3? [Y by default | n]') != 'n'
            run_full_experiment(TX_TRAINING_NODES, TX_TESTING_NODES, RX_NODES, AP_NODE, EXPERIMENT_DIR, epochs, 'probe', cloud_sync)
        elif instruction == 'run experiment udp':
            epochs = int(input('How many epochs? (int only please): '))
            cloud_sync = input('Should we record to AWS S3? [Y by default | n]') != 'n'
            run_full_experiment(TX_TRAINING_NODES, TX_TESTING_NODES, RX_NODES, AP_NODE, EXPERIMENT_DIR, epochs, 'udp', cloud_sync)
        else: print('Invalid command.')

if __name__ == "__main__":
    main()