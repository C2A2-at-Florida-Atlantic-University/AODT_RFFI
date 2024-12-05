import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt

def cfo_estimate(x, D):
    num_samples = len(x)

    unused = np.mod(num_samples, D)

    cx = x[0:int(len(x) - (D + unused))]
    sx = x[D:int(len(x)-unused)]

    res = np.dot(np.conj(cx), sx)

    foffset = np.angle(np.sum(res)) / (2 * np.pi)

    return foffset

def coarse_cfo_estimate(stf, fs):
    # Optional correlation offset
    corr_offset = 0.75
    fft_len = 64

    # Coarse CFO estimate assuming 4 repetitions per FFT period
    M = int(fft_len/4)     # Number of samples per repetition
    GI = fft_len/4    # Guard interval length
    S = M*9          # Maximum useful part of L-STF
    N = len(stf) # Number of samples in the input

    offset = np.round(corr_offset * GI)
    use_idx = [int(i) for i in offset + np.arange(min(S, N-offset))]
    use = stf[use_idx]
    foffset = cfo_estimate(use, M) * fs / M

    if abs(foffset) > fs/2:  # Nyquist limit
        print(f"CFO estimate ({foffset}) exceeds Nyquist frequency")

    return foffset

def fine_cfo_estimate(ltf, fs):
    # Optional correlation offset
    corr_offset = 0.75
    fft_len = 64

    # Fine CFO estimate assuming one repetition per FFT period (2 OFDM symbols)
    M = int(fft_len)      # Number of samples per repetition
    GI = fft_len/2    # Guard interval length
    S = M*2          # Maximum useful part of L-LTF (2 OFDM symbols)
    N = len(ltf) # Number of samples in the input

    offset = round(corr_offset * GI)
    use_idx = [int(i) for i in offset + np.arange(min(S, N-offset))]
    use = ltf[use_idx]
    foffset = cfo_estimate(use, M) * fs / M
    return foffset

def extract_preamble_cfo(preamble, show=False):
    # Downsample from 25 to 20 Msps
    preamble = signal.resample_poly(preamble, up=4, down=5)

    preamble_stf = preamble[0:160]
    cfo_coarse = coarse_cfo_estimate(preamble_stf, 20e6)
    preamble = preamble * np.exp(1j * (np.arange(1, len(preamble) + 1) / 20e6 * 2 * np.pi * (-cfo_coarse)))

    preamble_ltf = preamble[160:320]
    cfo_fine = fine_cfo_estimate(preamble_ltf, 20e6)
    preamble = preamble * np.exp(1j * (np.arange(1, len(preamble) + 1) / 20e6 * 2 * np.pi * (-cfo_fine)))

    # Upsample back from 20 to 25 Msps
    preamble = signal.resample_poly(preamble, up=5, down=4)

    if show:
        print(f"CFO coarse: {np.round(cfo_coarse, 2)}")
        print(f"CFO fine: {np.round(cfo_fine, 2)}")

    return cfo_coarse, cfo_fine

def extract_data_cfo(data):
    """
    Returns an array of CFO values (first column -- coarse, second column -- fine cfo).
    You can also add the two columns to obtain final CFO values.
    """
    cfo = np.zeros((data.shape[0], 2))
    for i in range(data.shape[0]):
        preamble = data[i, :].squeeze()
        cfo_coarse, cfo_fine = extract_preamble_cfo(preamble)
        cfo[i, 0] = cfo_coarse
        cfo[i, 1] = cfo_fine
    return cfo

def compensate_cfo(data, cfo):
    """
    Compensates the data for the given CFO values.
    """
    compensated_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        preamble_cfo = cfo[i]
        phase = np.arange(1, len(data[i, :]) + 1) / 20e6 * 2 * np.pi * (-1 * preamble_cfo)
        compensated_data[i, :] = data[i, :] * np.exp(1j * phase)
    return compensated_data

def generate_cfo_values(n_samples, distribution, ppm_range=[-40, 40], freq=2.4e9, rnd=np.random.default_rng(), show=False):
    """
    Generate n_samples CFO values using a given distribution.
    Accepted distribution types: 
    - uniform
    - gaussian
    """
    if n_samples <= 0 or distribution not in ['uniform', 'gaussian']:
        raise ValueError("Invalid input parameters")
    
    if distribution == 'uniform':
        dist = stats.uniform(loc=ppm_range[0], scale=abs(ppm_range[1] - ppm_range[0]))
    elif distribution == 'gaussian':
        mean = np.mean(ppm_range)
        two_sigma = 2 # less pointy :D 
        # two_sigma = 4
        # two_sigma = 6 # more pointy :D 
        dist = stats.truncnorm(
            a=(ppm_range[0] - mean) / (abs(ppm_range[1] - ppm_range[0])/two_sigma),
            b=(ppm_range[1] - mean) / (abs(ppm_range[1] - ppm_range[0])/two_sigma),
            loc=mean,
            scale=(abs(ppm_range[1] - ppm_range[0])/two_sigma))
        
    ppm_values = dist.rvs(size=n_samples, random_state=rnd)
    cfo_hz = np.array(ppm_values * 1e-6 * freq)

    if show:
        plt.figure(figsize=(10, 8), dpi=80)
        plt.hist(cfo_hz / 1e3, bins=100)
        plt.ylabel('Sample Frequency')
        plt.xlabel('CFO, kHz')
        plt.show()

    return cfo_hz
