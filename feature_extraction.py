import pickle
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
import pandas as pd


def extract_features(file_name):
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    print((data.keys()))
    array_of_data = []
    array_of_signal_types = []

    columnlabels = ["signal_type", "snr", "magnitude_mean", "magnitude_std", "magnitude_skew", "magnitude_kurtosis", "phase_mean", "phase_std", "phase_skew", "phase_kurtosis", "spectral_entropy", "peak_frequency", "average_power"]
    for key in data.keys():
        signal_type = key[0]
        # print(signal_type)
        array_of_signal_types.append(signal_type)
        if signal_type in ['AM-SSB', 'WBFM', 'BPSK', 'CPFSK', 'GFSK', 'AM-DSB', 'QAM16', 'QAM64', 'QPSK', '8PSK', 'PAM4']:
            snr = key[1]
            if int(snr) <= -100:
                samples = data[key]
                for i in range(len(samples)):
                    i_samples = samples[i][0]
                    q_samples = samples[i][1]
                    magnitude = np.sqrt(i_samples**2 + q_samples**2)
                    phase = np.arctan2(q_samples, i_samples)

                    # Calculate statistical features for magnitude
                    magnitude_mean = np.mean(magnitude)
                    magnitude_std = np.std(magnitude)
                    magnitude_skew = skew(magnitude)
                    magnitude_kurtosis = kurtosis(magnitude)

                    # Calculate statistical features for phase
                    phase_mean = np.mean(phase)
                    phase_std = np.std(phase)
                    phase_skew = skew(phase)
                    phase_kurtosis = kurtosis(phase)

                    frequencies, psd = welch(magnitude, nperseg=128)
                    psd_norm = psd / np.sum(psd)
                    spectral_entropy = entropy(psd_norm)

                    peak_frequency = frequencies[np.argmax(psd)]

                    average_power = np.mean(psd)
                    array_of_data.append([signal_type, snr, magnitude_mean, magnitude_std, magnitude_skew, magnitude_kurtosis, phase_mean, phase_std, phase_skew, phase_kurtosis, spectral_entropy, peak_frequency, average_power])
    df = pd.DataFrame(array_of_data, columns = columnlabels)
    # df.to_csv("extracted_features_5_types_snr_gt_zero.csv", index=False)
    unique_signal_types = set(array_of_signal_types)
    print(unique_signal_types)
    return df