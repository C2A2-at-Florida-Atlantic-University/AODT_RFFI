# Dataset Capture Tools

This directory contains scripts to automate data capture for our paper. You can either access the data we captured directly, or replicate the capture on your own, according to your requirements. If you need to follow the process on your own, keep reading.

## The Testbed: 

These scripts perform automated data capture in the [Orbit testbed](https://orbit-lab.org/) at Rutgers University. Since we need to evaluate a device fingerprinting method, we need to collect IQ samples emitted by many devices, and capture them simultaneously on several receivers. 

We use N210r4 USRPs as our receivers, configured to capture samples on 11th channel @ 2.4 GHz with 25 Msps sampling rate. Our transmitters aren't sending signal to USRPs directly; intead each of them (one-by-one) is connected to an access point from the outdoor environment, and we emit UDP packets with random data for 2 seconds. 

## How Does It Work?

For detailed description, please refer to our paper. In short, there are several automation scripts included in this repository:

* [rx_master.py](rx_master.py): this script can be used to configure a given node as a receiver, and capture a single TX-RX transmission in a .bin file. This script also downloads the bin file with IQ samples into a specified directory.

* [tx_udp_master.py](tx_udp_master.py): this script can be used to configure a TX and establish a connection between TX and AP, start and stop UDP traffic emission/capture, disconnect the TX, and put it back into rfkill mode. 

* [master.py](master.py): this is our main automation script. It is built for two tasks. First, to configure a series of specified transmitters, receivers an and an access point. Second, to perform continuous data capture for a specified number of epochs (also named 'rounds' in the paper). A 'round' is an event during which signal is transmitted once from each of the pre-defined subset of devices. We conduct many of these rounds during our experiments to evaluate fingerprint stability.

## How to Run:

### Important preliminaries: 

You can run this code on your local machine, or on a remote server. We do recommend running it on a remote VM, if you expect it to run for an extended periods of time for several reasons:

1. Usually remote VMs provide better internet bandwidth, meaning that your .bin files will be downloaded from Orbit nodes (and uploaded to AWS S3) much faster;

2. You don't risk failing an experiment due to a lost connection, etc;

3. Even though we store raw data on AWS S3, you will need to use your local device storage as an intermediary place for data captured during an ongoing epoch (aka 'round'). 

### Step 1. Reserve Orbit environments

Go to [Orbit Lab website](https://orbit-lab.org/), sign up and reserve two Orbit environments: `grid` and `outdoor`. 

### Step 2. Configure the experiment

1. Configure the [.env](.env) file (only required for the Master experiment): 
    * `OPENAI_API_KEY`: OpenAI API key for assisting with SSH command generation;
    * `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`: AWS credentials for uploading data to S3;
    
2. Configure the TX settings:
    * `TX_CHANNEL`: WiFi channel on which you want to transmit data
    * `JUMP_NODE_GRID`: credentials & route to the grid server for accessing the grid nodes
    * `JUMP_NODE_OUTDOOR`: credentials & route to the grid server for accessing the outdoor nodes

3. Configure the RX settings: 
    * `JUMP_NODE_GRID`: credentials & route to the grid server for accessing the grid nodes
    * `EXPERIMENT_DIR`: your local directory to store your .bin files
    * `RX_CHANNEL_IDX`: WiFi channel on which you want to transmit data
    * `RX_GAIN`: gain in dB of the USRP receiver
    * `RX_SAMP_RATE`: sampling rate for the USRP receiver (should be at least 20 Msps for 2.4 GHz WiFi)

4. Configure the Master settings: 
    * `AP_NODE`: name of the AP in the outdoor environment (i.e., node2-5)
    * `RX_NODES`: list of RX nodes in the grid environment with USRPs on board
    * `TX_TRAINING_NODES`: list of TX nodes in the grid environment for capturing the training dataset (one epoch)
    * `TX_TESTING_NODES`: list of TX nodes in the grid environment for capturing the testing dataset (multiple epochs)
    * `TX_CHANNEL`: WiFi channel on which you want to transmit data
    * `RX_CAP_LEN_UDP`: how many seconds of data to capture on the USRP
    * `CONFIG_BATCH_SIZE`: how many configuration threads can be run in parallel
    * `AWS_S3_BUCKET_NAME`: name of the AWS bucket where to store the captured epochs (the script generates epoch dirs automatically)
    * `EXPERIMENT_DIR`: your local directory to store your .bin files

5. To run the code for a single TX/RX transmission:

    1. Install the [requirements](requirements.txt) (we recommend using `Conda`): `pip3 install -r requirements.txt`
    2. Open two separate terminal sessions: one for TX, another for RX connection;
    3. Launch the RX script, and follow the internal questions: `python3 rx_master.py`
    4. Launch the TX script, and follow the internal questions: `python3 tx_udp_master.py`

6. To run the code for a long-term experiment:

    1. Install the [requirements](requirements.txt) (we recommend using `Conda`): `pip3 install -r requirements.txt`
    2. Launch the master script, and follow the internal questions: `python3 master.py`
        * Run sensor configurations;
        * Manually verify that each device has been properly configured (sometimes, Orbit testbed devices may require custom configurations due to experiments by other parties);
        * Launch data capture (this process will first, run capture of a single epoch from training devices, and then a specified number of epochs from testing devices).

7. Done!