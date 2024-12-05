from torch import normal
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import pickle
import hashlib
from IPython.display import display
from functools import reduce
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

def intersect_n(*arrays):
    return list(reduce(np.intersect1d, arrays))

def read_dat_iq_file(file_path):
    raw_data = np.fromfile(file_path, dtype=np.float32)
    
    iq_samples = raw_data[::2] + 1j * raw_data[1::2]
    
    return iq_samples

def request_value_dropdown(prompt, options, callback):
    dropdown = widgets.Dropdown(
        options=options,
        value=options[0],
        description=prompt,
    )

    # Function to handle the dropdown value change
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            print(f"Selected option: {change['new']}")
            callback(change['new'])

    # Attach the function to the dropdown
    dropdown.observe(on_change)

    # Display the dropdown
    display(dropdown)

def generate_grid_node_ids():
    ids = {}
    coordinates = {}
    node_i = 0
    for i in np.arange(1, 21):
        for j in np.arange(1, 21):
            ids[str(i) + "-" + str(j)] = node_i
            node_i = node_i + 1
            coordinates[node_i] = (i, j)
    return ids, coordinates

def apply_ieee_style():
    plt.style.use('default')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "text.usetex": False,
        "axes.linewidth": 1,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1,

        # Grayscale settings
        "image.cmap": "gray",
        "axes.facecolor": 'white',
        "figure.facecolor": 'white',
        "figure.edgecolor": 'white',
        "savefig.facecolor": 'white',
        "savefig.edgecolor": 'white',
        "grid.color": '0.8',
        "text.color": 'black',
        "axes.edgecolor": 'black',
        "axes.labelcolor": 'black',
        "xtick.color": 'black',
        "ytick.color": 'black'
    })

def extract_unix_timestamp_ms(file_path):
    from datetime import datetime
    
    # Extract the timestamp part from the file path
    timestamp_str = file_path.split('_epoch_')[1].replace('.h5', '')
    
    # Convert the timestamp string to a datetime object
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
    
    # Convert the datetime object to unix time in milliseconds
    unix_timestamp_ms = int(timestamp.timestamp() * 1000)
    
    return unix_timestamp_ms

def convert_ms_to_time_label(unix_timestamp_ms):
    # Convert milliseconds to seconds, minutes, and hours
    total_seconds = unix_timestamp_ms // 1000
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    seconds = total_seconds % 60
    
    # Create a time label based on the length of time
    if hours > 0:
        return f'{hours}h'
    elif minutes > 0:
        return f'{minutes}m'
    else:
        return f'{seconds}s'

def plot_roc_curves(roc_curves, figtitle=None):
    lw = 4
    apply_ieee_style()
    plt.figure(figsize=(9, 10), dpi=300)
    plt.rcParams.update({"font.size": 30})
    for roc_curve in roc_curves:
        fpr = roc_curve['fpr']
        tpr = roc_curve['tpr']
        label = roc_curve['info']
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC: {auc_score*100:.2f}%', linewidth=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if figtitle is not None: plt.title(figtitle)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def hash_object(obj):
    # Serialize the object with pickle
    obj_bytes = pickle.dumps(obj)
    # Hash the serialized data
    hash_object = hashlib.sha256(obj_bytes)
    return hash_object.hexdigest()

def calculate_preamble_rssi(iq_samples):
    """
    Calculate RSSI from a vector of complex IQ samples.
    
    Parameters:
    iq_samples (np.ndarray): Array of complex IQ samples.
    
    Returns:
    float: Estimated RSSI in dB.
    """
    # Step 1: Calculate power for each IQ sample
    power_samples = np.abs(iq_samples) ** 2
    
    # Step 2: Average the power
    avg_power = np.mean(power_samples)
    
    # Step 3: Convert to dB
    rssi_db = 10 * np.log10(avg_power)
    
    return rssi_db

def filter_abnormal_cfo(cfo_values, plot=False):
    q1 = np.percentile(cfo_values, 10)
    q3 = np.percentile(cfo_values, 90)

    iqr = abs(q3 - q1)

    sensitivity = 0

    lower_bound = q1 - sensitivity * iqr
    upper_bound = q3 + sensitivity * iqr
    normal_indices = np.where((cfo_values >= lower_bound) & (cfo_values <= upper_bound))[0]

    if plot:
        print(f"Frames retained: {len(normal_indices)} / {len(cfo_values)}")
        plt.figure(figsize=(10, 6), dpi=120)
        plt.hist(cfo_values, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(lower_bound, color='black', linestyle='--', label=f'Lower Bound ({lower_bound:.2f} dB)')
        plt.axvline(upper_bound, color='red', linestyle='--', label=f'Upper Bound ({upper_bound:.2f} dB)')
        plt.title(f'CFO Distribution with Outlier Bounds.')
        plt.xlabel('CFO (Hz)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return normal_indices

def filter_abnormal_rssi(rssi_values, plot=False):
    """
    Filters out frames with abnormal RSSI values based on the IQR method,
    with adjustable sensitivity.
    
    Parameters:
    rssi_values (list or np.ndarray): List or array of RSSI values.
    sensitivity (float): Sensitivity level between 0 and 1; higher values
                         make the function stricter and remove more data.
    
    Returns:
    np.ndarray: Indices of frames considered normal based on RSSI.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1 = np.percentile(rssi_values, 25)
    q3 = np.percentile(rssi_values, 75)

    # Calculate the IQR
    iqr = abs(q3 - q1)

    # Sensitivity
    sensitivity = 0
    
    # Define lower and upper bounds
    lower_bound = q1 - sensitivity * iqr
    upper_bound = q3 + sensitivity * iqr
    
    # Get indices of normal frames
    normal_indices = np.where((rssi_values >= lower_bound) & (rssi_values <= upper_bound))[0]

    if plot:
        print(f"Frames retained: {len(normal_indices)} / {len(rssi_values)}")
        plt.figure(figsize=(10, 6), dpi=50)
        plt.hist(rssi_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(lower_bound, color='red', linestyle='--', label=f'Lower Bound ({lower_bound:.2f} dB)')
        plt.axvline(upper_bound, color='red', linestyle='--', label=f'Upper Bound ({upper_bound:.2f} dB)')
        plt.title(f'RSSI Distribution with Outlier Bounds.')
        plt.xlabel('RSSI (dB)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    
    return normal_indices