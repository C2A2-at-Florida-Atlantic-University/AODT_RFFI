# Dataset Preprocessing Tools

This directory contains scripts for transforming captured raw IQ samples into pickle files that can be used for training the extractor model and testing the fingerprinting system.

For details about this code, please refer to our paper. 

Importantly, even though this code is written in Python, it relies on Matlab scripts for decoding OFDM frames. Therefore, please ensure that you have the latest version of Matlab with an active license installed on your device. For this paper, we used the Matlab version R2024a.

## How to Run:

On macOS: 

1. Ensure that you can launch matlab in Terminal: `alias matlab="/Applications/MATLAB_R2024a.app/bin/matlab"`

    * Optionally, add it to ~/.bash_profile, and then run `source ~/.bash_profile` 

2. In multiple terminal windows, open N matlab sessions: `matlab -nodesktop -r "matlab.engine.shareEngine('mobintel_session_X')"`

    * X would need to be changed to the index of your session
    * Start as many sessions as you specified in the `MATLAB_SESSION_NAMES` list in the script

3. Configure the script:

    * `ROOT_DIR`: this is a root directory where you'll store the directory with this code, as well as the temporary IQ data dir;
    * `MATLAB_OFDM_DECODER`: path to current codebase directory;
    * `TEMP_IQ_DIRECTORY`: directory to temporarily store .bin files during the pre-processing routine;
    * `NODE_MACS`: specifies name of the JSON file, stored in the RFFI_DATASET_TARGET_DIR, containing node metadata from S3 bucket (experiment_device_macs.json);
    * `S3_BUCKET_NAME`: name of the S3 bucket where all experiments are stored;
    * `S3_EXPERIMENT_NAME`: name of the S3 sub-bucket which specifies which experiment to process;
    * `FRAME_COUNT`: specifies how many frames in each .bin file we need to extract (if there aren't enough frames available - this .bin file will be skipped);
    * `COMPLETED_SESSIONS`: this is an array that you can keep empty, unless you ran this task before and have sessions you don't want to process again;
    * `MATLAB_SESSION_NAMES`: this is a list of Matlab session names from the previous step.

4. Launch the pre-processing script: `python3 prepare.py`
    * Specify the preamble length: `400 for 25 Msps, 320 for 20 Msps`

5. Done! Please note that this script can take a long time to process. Additionally, if you download data from S3, consider the costs associated with transfering the data to your device, as this task may incur significant costs.