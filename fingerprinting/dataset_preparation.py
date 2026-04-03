import numpy as np
import h5py
try:
    import seaborn as sea  # noqa: F401
except ImportError:
    sea = None
import matplotlib.pyplot as plt
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform
from scipy import signal

np.random.seed(42)

def awgn(data, snr_range):
    
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0],snr_range[-1],pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        SNR_linear = 10**(SNRdB[pktIdx]/10)
        P= sum(abs(s)**2)/len(s)
        N0=P/SNR_linear
        n = sqrt(N0/2)*(standard_normal(len(s))+1j*standard_normal(len(s)))
        data[pktIdx] = s + n

    return data 

class LoadDataset():
    def __init__(self,):
        self.dataset_name = 'data'
        self.labelset_name = 'label'
        self.rssiset_name = 'rssi'

    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        return data[:, 0::2] + 1j * data[:, 1::2]

    def shuffle(self, data, labels):
        # Produce a new order for elements
        new_order = np.arange(labels.shape[0])
        np.random.shuffle(new_order)

        data = data[new_order, :]
        labels = labels[new_order]

        return data, labels
    
    def load_iq_samples(self, file_path):
        '''
        Load IQ samples from a dataset.
        
        INPUT:
            FILE_PATH is the dataset path.
            
            DEV_RANGE specifies the loaded device range.
            
            PKT_RANGE specifies the loaded packets range.
            
        RETURN:
            DATA is the laoded complex IQ samples.
            
            LABLE is the true label of each received packet.
        '''
        
        f = h5py.File(file_path,'r')
        rssi = f[self.rssiset_name][:]
        label = f[self.labelset_name][:]
        label = label.astype(int)

        # # If the list of devices isn't specified - loop through all of the available ones
        # if dev_range is None:
        #     dev_range = set(label.flatten())

        # # Filter indexes of frames to keep based on dev_range
        # frame_idx_filtered = []
        # for dev_idx in dev_range:
        #     # Extract only the specified devices, and for each only pkt_range frames
        #     frame_idx_device = np.where(label==int(dev_idx))[0][pkt_range]
        #     frame_idx_filtered.extend(frame_idx_device)
    
        # Retrieve data from the dataset
        data = f[self.dataset_name][:]

        # Convert from interleaved doubles to complex values
        data = self._convert_to_complex(data)
        
        # # Filter the dataset based on dev_range and pkt_range
        # label = label[frame_idx_filtered]
        # data = data[frame_idx_filtered, :]
          
        f.close()
        return data, label, rssi

class ChannelIndSpectrogram():
    def __init__(self,):
        pass
    
    def _normalization(self, data):
        data_normalized = np.zeros(data.shape, dtype=complex)
        for i in range(data.shape[0]):
            data_normalized[i] = data[i] / np.sqrt(np.mean(np.abs(data[i])**2))
        return data_normalized        

    def _channel_ind_spectrogram_single(self, frame, win_len, overlap, enable_ind=True):
        _, t, spec = signal.stft(frame, window = 'boxcar', 
                                nperseg = win_len, noverlap = overlap, 
                                nfft = win_len, return_onesided = False, 
                                padded = False, boundary = None)
        spec = np.fft.fftshift(spec, axes=0)

        # If enabled, produce channel-independent spectrogram
        if enable_ind:
            spec = spec[:, 1:] / spec[:, :-1]

        # Return logarithm of the spectrogram magnitude
        spec = np.log10(np.abs(spec)**2)

        # NEW: Apply standardization to obtain more spectrogram consistency
        spec = self._standardization(spec)

        return t, spec

    def _standardization(self, spec):
        mean = spec.mean()
        std = spec.std()
        spec = (spec - mean) / std
        return spec

    def channel_ind_spectrogram(self, data, row, enable_ind, overlap_coef = 0.9, remove_subcarriers=True, return_spec_t=False):
        # Normalize IQ samples
        data = self._normalization(data)

        overlap = row * overlap_coef

        # Produce spectrogram once to dynamically determine input array dimensions
        t, test_run = self._channel_ind_spectrogram_single(data[0], win_len=row, overlap=overlap, enable_ind=enable_ind)

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        data_spectrograms = np.zeros([data.shape[0], test_run.shape[0], test_run.shape[1], 1])

        # Run STFT for each frame separately
        for i in np.arange(data.shape[0]):
            _, spec = self._channel_ind_spectrogram_single(data[i], win_len=row, overlap=overlap, enable_ind=enable_ind)
            data_spectrograms[i,:,:,0] = spec

        if remove_subcarriers:
            freq_bins = data_spectrograms.shape[1]
            if freq_bins != 80:
                raise ValueError(
                    f"Guard subcarrier removal assumes row=80 (got freq_bins={freq_bins}). "
                    "Set remove_subcarriers=False or update guard indices for this row size."
                )
            guards = list(range(0, 14)) + [40] + list(range(67, 80))
            data_spectrograms = np.delete(data_spectrograms, guards, axis=1)

        if return_spec_t: return data_spectrograms, t
        else: return data_spectrograms