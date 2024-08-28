import os
import time
import subprocess
from openai_client import OpenAIClient

# Channel Number:   1         2         3         4         5         6         7         8         9         10        11        12        13
OFDM_CENTER_FREQ = ["2412e6", "2417e6", "2422e6", "2427e6", "2432e6", "2437e6", "2442e6", "2447e6", "2452e6", "2457e6", "2462e6", "2467e6", "2472e6"]
EXPERIMENT_DIR = "/Users/stepanmazokha/Desktop/orbit_dataset"
CORE_RX_FILE = "/root/samples.dat"
RX_CHANNEL_IDX = 11 # 1-based value, according to 802.11 standard
RX_USRP_IP = "addr=192.168.10.2"
RX_FREQ = OFDM_CENTER_FREQ[RX_CHANNEL_IDX - 1]
RX_GAIN = "10" # Chx Gain Value, Absolute (dB), range (for SBX): 0 - 31.5 dB
RX_SAMP_RATE = "25e6" # Sampling rate, should be at least 20 Msps
# RX_SAMP_RATE = "20e6" # Sampling rate, should be at least 20 Msps
RX_SKIP = "0" # How many samples (N) do we skip, where N = RX_SKIP * RX_SAMP_RATE
RX_CAP_LEN = "10" # For how long do we capture samples (in seconds)
RX_LO_OFF = "0" # If the center freq is crowded, we can optionally tune it up (WiSig had it at 10 MHz)
LLM_MAX_ATTEMPTS = 6 # How many times we'll use LLM to attempt node connection

JUMP_NODE_GRID = "smazokha@grid.orbit-lab.org"

def generate_dir_name():
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def prepare_target_dir(experiment_dir, prefix):
    return os.path.join(experiment_dir, prefix + generate_dir_name())

def send_command(needsJump, node, command, capture_response=False):
    if needsJump: 
        cmd = "ssh -o StrictHostKeyChecking=no -J %s root@%s \"%s\"" % (JUMP_NODE_GRID, node, command)
    else: 
        cmd = "ssh -o StrictHostKeyChecking=no %s \"%s\"" % (node, command)

    print(f"[{node}] {cmd}")
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout_lines = []

    while True:
        stdout_line = process.stdout.readline()

        if not stdout_line: 
            break

        if stdout_line:
            print(f"[{node}] {stdout_line}", end='')
            if capture_response:
                stdout_lines.append(stdout_line)
        
    if capture_response:
        stdout, _ = process.communicate()
        stdout_lines.append(stdout)    
        return ''.join(stdout_lines)
    else: return None

def node_ready_wait(node_id, wait_seconds=30):
    attempts = 0
    while attempts < LLM_MAX_ATTEMPTS:
        print(f'Sleeping for {wait_seconds} seconds before attempting to connect.')
        time.sleep(wait_seconds)
        
        attempts += 1

        can_proceed = OpenAIClient().prompt_is_ls_successful(send_command(True, node_id, "ls /root/", capture_response=True))

        if can_proceed:
            break
        
        if attempts == LLM_MAX_ATTEMPTS:
            print(f"[{node_id}] This was the last attempt. Node is dead. Quitting.")
            return

def node_configure(node_id):
    openai_client = OpenAIClient()

    send_command(False, JUMP_NODE_GRID, "omf tell -a offh -t " + node_id)
    send_command(False, JUMP_NODE_GRID, "omf load -i baseline.ndz -t " + node_id)
    send_command(False, JUMP_NODE_GRID, "omf tell -a on -t " + node_id)

    node_ready_wait(node_id, wait_seconds=30)

    send_command(True, node_id, "sudo add-apt-repository ppa:gnuradio/gnuradio-releases -y")
    send_command(True, node_id, "sudo apt update -y")
    send_command(True, node_id, "sudo apt-get update -y")
    send_command(True, node_id, "sudo apt-get install -y uhd-host net-tools wireless-tools git python3-pip gnuradio gir1.2-gtk-3.0 rfkill")
    send_command(True, node_id, "rfkill block wlan")

    stdout = send_command(True, node_id, "ifconfig", capture_response=True)
    if stdout.__contains__('DATA2'):
        usrp_interface = 'DATA2'
    else:
        usrp_interface = openai_client.prompt_find_usrp_interface(stdout)

    # if usrp_interface != 'NONE':
    send_command(True, node_id, f"ifconfig {usrp_interface} 192.168.10.1 netmask 255.255.255.0 up")
    # else:
    #     while True:
    #         interface = input("Which interface should we use?")
    #         send_command(True, node_id, f"ifconfig {interface} 192.168.10.1 netmask 255.255.255.0 up")
    #         send_command(True, node_id, "uhd_find_devices")

    #         instruction = input("Did it work? [Y/any key]")
    #         if instruction == 'Y': break
    #         else: continue

    send_command(True, node_id, f'/usr/lib/uhd/examples/test_pps_input --args=\"{RX_USRP_IP}\" --source external')

    send_command(True, node_id, "cd /root/ && git clone https://github.com/i-sense/mobintel-rffi && mv /root/mobintel-rffi/orbit-capture/gnuradio-n210 /root/ && rm -rf /root/mobintel-rffi")

    # These commands take a long time to run, but are paramount for reducing background noise, etc
    send_command(True, node_id, 'uhd_cal_rx_iq_balance --verbose --args="addr=192.168.10.2"')
    # send_command(True, node_id, 'uhd_cal_tx_iq_balance --verbose --args="addr=192.168.10.2"')
    # send_command(True, node_id, 'uhd_cal_tx_dc_offset --verbose --args="addr=192.168.10.2"')

    print(f'RX node {node_id} configured.')

def node_capture(tx_node_id, rx_node_id, target_dir, cap_len_sec):
    # 0. Remove any residual files, if any
    send_command(True, rx_node_id, f"rm -rf {CORE_RX_FILE}")

    # 1. Launch capture
    send_command(True, rx_node_id, f'/root/gnuradio-n210/receive_capture.py --device=\"{RX_USRP_IP}\" --cap-len={cap_len_sec} --output-file=\"{CORE_RX_FILE}\" --rx-freq={RX_FREQ} --rx-gain={RX_GAIN} --rx-lo-off={RX_LO_OFF} --rx-samp-rate={RX_SAMP_RATE} --skip={RX_SKIP}')

    # 2. Download file to local device
    filename = f"tx{{node_{tx_node_id}}}_rx{{node_{rx_node_id}+rxFreq_{RX_FREQ}+rxGain_{RX_GAIN}+capLen_{cap_len_sec}+rxSampRate_{RX_SAMP_RATE}}}.dat"
    path_local = os.path.join(target_dir, filename)
    command = f"scp -o StrictHostKeyChecking=no -J {JUMP_NODE_GRID} root@{rx_node_id}:{CORE_RX_FILE} {path_local}"
    print(command)
    os.system(command)

    # 3. Delete file on the node
    send_command(True, rx_node_id, f"rm -rf {CORE_RX_FILE}")
    print(f'Capture completed for TX {tx_node_id}].')

    return path_local

def mode_rx(node_ids):
    if len(node_ids) == 0:
        print('No nodes to emit from.')
        return

    # Ensure that the dir for the experiment is correct
    input(f"Experiment dir: {EXPERIMENT_DIR}. OK?")
    experiment_dir = EXPERIMENT_DIR
    os.makedirs(experiment_dir, exist_ok=True)

    # Generate & create directory for the RX epoch
    target_folder = prepare_target_dir(experiment_dir, 'epoch_')
    os.mkdir(target_folder)
    print(f"Will store files here: {target_folder}")
    
    tx_node_id = input("TX node ID: ")

    node_idx = 0
    while node_idx < len(node_ids):
        rx_node_id = node_ids[node_idx]

        instruction = input(f"Ready to RX on {rx_node_id}? [Y/skip/done]")

        if instruction == 'Y':
            node_capture(tx_node_id, rx_node_id, target_folder, RX_CAP_LEN)
            node_idx = node_idx + 1
        elif instruction == 'skip':
            node_idx = node_idx + 1
        elif instruction == 'done':
            break
        else: print('Invalid command')

    print('Done')

def mode_config(node_ids):
    if len(node_ids) == 0:
        print('No nodes to emit from.')
        return
    
    node_idx = 0

    while node_idx < len(node_ids):
        node_id = node_ids[node_idx]

        instruction = input(f"Ready to configure {node_id}? [Y/skip/done]")

        if instruction == 'Y':
            node_configure(node_id)
            node_idx = node_idx + 1
        elif instruction == 'skip':
            node_idx = node_idx + 1
        elif instruction == 'done':
            break
        else: print('Invalid command')

    print('Done')

def main():
    print("Welcome! Let's get started.")

    os.system(f"ssh -o StrictHostKeyChecking=no {JUMP_NODE_GRID} rm -rf ~/.ssh/known_hosts")

    while True:
        instruction = input("What should we do? [config one | rx one]")

        if instruction == 'config one':
            node_id = input("RX node ID: ")
            mode_config([node_id])
        elif instruction == 'rx one':
            node_id = input('RX node ID: ')
            mode_rx([node_id])
        else: print("Wrong command.")

if __name__ == "__main__":
    main()