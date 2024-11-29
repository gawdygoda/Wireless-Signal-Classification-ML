import time
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.ml_wireless_classification.base.SignalUtils import (
    compute_instantaneous_features, compute_modulation_index, compute_spectral_asymmetry,
    instantaneous_frequency_deviation, spectral_entropy, envelope_mean_variance,
    spectral_flatness, spectral_peaks_bandwidth, zero_crossing_rate, compute_fft_features,
    autocorrelation, is_digital_signal, compute_kurtosis, compute_skewness,
    compute_spectral_energy_concentration, compute_instantaneous_frequency_jitter,
    compute_spectral_kurtosis, compute_higher_order_cumulants, compute_crest_factor,
    compute_spectral_entropy, compute_energy_spread, compute_autocorrelation_decay,
    compute_rms_of_instantaneous_frequency, compute_entropy_of_instantaneous_frequency,
    compute_envelope_variance, compute_papr
)
import joblib
from joblib import parallel_backend

DIRECTORY = "./data"
FILE_NAME = "RML2016.10a_dict.pkl"
MODEL_TYPE = "SVC_RBF" #SVC_LINEAR, SVC_POLY, SVC_RBF, RND_FOREST
SAVE_PLOTS_FLAG = 1
feature_dict = {} # Global dictionary to store feature names and values

# n_jobs=-2 means run on all CPUs - 1 (leave one for me to surf the web!)
with parallel_backend('threading', n_jobs=-2):

    def plot_all_analysis(X_test, y_test, label_encoder, show_plot=False):
        """
        Combines all analysis and visualization into a single function for a single-branch model.
        Organizes the plots in a single figure with a dynamic grid layout.
        Saves the figure to the specified directory with the model name as the file name.
        Returns a dictionary of statistics such as accuracy over 5 dB and accuracy per SNR.
        """
        # Get the directory of the current script
        #script_dir = os.path.dirname(os.path.abspath(__file__))
        #common_vars.stats_dir = os.path.join(script_dir, "..", "stats")

        # Load the trained model from a file
        model_file_name = f"model_{MODEL_TYPE}.pkl"
        model_path = os.path.join(DIRECTORY, model_file_name)
        model = joblib.load(model_path)

        # --- Confusion Matrix for All SNR Levels ---
        y_pred = model.predict(X_test)

        elapsed_time = time.time() - start_time
        print(f"Full Prediction Done...{elapsed_time:.3f} seconds")

        conf_matrix = confusion_matrix(y_test, y_pred)
        overall_accuracy = accuracy_score(y_test, y_pred) * 100

        elapsed_time = time.time() - start_time
        print(f"Full Prediction Conf Matrix and Accuracy Done...{elapsed_time:.3f} seconds")

        scaled_SNR_5 = 0.52

        # --- Confusion Matrix for SNR > 5 dB ---
        snr_above_5_indices = np.where(X_test[:, -1] > scaled_SNR_5) #np.where(X_test[:, :, 2].mean(axis=1) > 5)
        X_test_snr_above_5 = X_test[snr_above_5_indices]
        y_test_snr_above_5 = y_test[snr_above_5_indices]

        if len(X_test_snr_above_5) > 0:
            y_pred_snr_above_5 = y_pred[snr_above_5_indices]
            conf_matrix_snr_above_5 = confusion_matrix(y_test_snr_above_5, y_pred_snr_above_5)
            accuracy_over_5dB = accuracy_score(y_test_snr_above_5, y_pred_snr_above_5) * 100
        else:
            conf_matrix_snr_above_5 = None
            accuracy_over_5dB = None

        elapsed_time = time.time() - start_time
        print(f"SNR > 5 Prediction & Conf Matrix and Accuracy Done...{elapsed_time:.3f} seconds")

        # --- Accuracy vs. SNR ---
        unique_snrs = sorted(set(X_test[:, -1])) #sorted(set(X_test[:, :, 2].mean(axis=1)))
        accuracy_per_snr = []
        for snr in unique_snrs:
            snr_indices = np.where(X_test[:, -1] == snr) #np.where(X_test[:, :, 2].mean(axis=1) == snr)
            X_snr = X_test[snr_indices]
            y_snr = y_test[snr_indices]
            if len(y_snr) > 0:
                y_pred_snr = y_pred[snr_indices]
                accuracy_per_snr.append(accuracy_score(y_snr, y_pred_snr) * 100)
            else:
                accuracy_per_snr.append(np.nan)

        peak_accuracy = max([acc for acc in accuracy_per_snr if not np.isnan(acc)])
        peak_snr = unique_snrs[accuracy_per_snr.index(peak_accuracy)]

        elapsed_time = time.time() - start_time
        print(f"Accuracy vs SNR Prediction & Conf Matrix and Accuracy Done...{elapsed_time:.3f} seconds")

        # --- Accuracy vs. SNR per Modulation Type ---
        unique_modulations = label_encoder.classes_
        modulation_traces = []
        for mod_index, mod in enumerate(unique_modulations):
            accuracies = []
            for snr in unique_snrs:
                mod_snr_indices = np.where(
                    (y_test == mod_index) & (X_test[:, -1] == snr) #(X_test[:, :, 2].mean(axis=1) == snr)
                )
                X_mod_snr = X_test[mod_snr_indices]
                y_mod_snr = y_test[mod_snr_indices]
                if len(y_mod_snr) > 0:
                    y_pred_mod_snr = y_pred[mod_snr_indices]
                    accuracies.append(accuracy_score(y_mod_snr, y_pred_mod_snr) * 100)
                else:
                    accuracies.append(np.nan)
            modulation_traces.append((mod, accuracies))

        elapsed_time = time.time() - start_time
        print(f"SNR VS MOD Prediction & Conf Matrix and Accuracy Done...{elapsed_time:.3f} seconds")

        #convert unique_snrs & peak_snr back to unscaled transforms for plotting
        SNR_min = -20
        SNR_max = 18
        scaled_SNR = [round((x - min(unique_snrs)) / (max(unique_snrs) - min(unique_snrs)) * (SNR_max - SNR_min) + SNR_min) for x in unique_snrs]
        peak_snr = round((peak_snr - min(unique_snrs)) / (max(unique_snrs) - min(unique_snrs)) * (SNR_max - SNR_min) + SNR_min)
        unique_snrs = scaled_SNR

        # --- Create Subplots ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot Confusion Matrix for All SNR Levels
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[0, 0])
        axes[0, 0].set_title("Confusion Matrix (All SNR Levels)")
        axes[0, 0].set_xlabel("Predicted Label")
        axes[0, 0].set_ylabel("True Label")

        # Plot Confusion Matrix for SNR > 5 dB
        if conf_matrix_snr_above_5 is not None:
            sns.heatmap(conf_matrix_snr_above_5, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=axes[0, 1])
            axes[0, 1].set_title("Confusion Matrix (SNR > 5 dB)")
            axes[0, 1].set_xlabel("Predicted Label")
            axes[0, 1].set_ylabel("True Label")
        else:
            axes[0, 1].text(0.5, 0.5, "No Samples with SNR > 5 dB",
                            ha='center', va='center', fontsize=12)
            axes[0, 1].set_title("Confusion Matrix (SNR > 5 dB)")

        # Plot Accuracy vs. SNR
        axes[1, 0].plot(unique_snrs, accuracy_per_snr, 'b-o', label='Recognition Accuracy')
        axes[1, 0].plot(peak_snr, peak_accuracy, 'ro')  # Mark peak accuracy
        axes[1, 0].text(peak_snr, peak_accuracy + 1, f"{peak_accuracy:.2f}%",
                        ha='center', va='bottom', fontsize=10,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        axes[1, 0].set_title("Recognition Accuracy vs. SNR")
        axes[1, 0].set_xlabel("SNR (dB)")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim(0,100)

        # Plot Accuracy vs. SNR per Modulation Type
        for mod, accuracies in modulation_traces:
            axes[1, 1].plot(unique_snrs, accuracies, '-o', label=mod)
        axes[1, 1].set_title("Accuracy vs. SNR per Modulation Type")
        axes[1, 1].set_xlabel("SNR (dB)")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].legend(loc='upper left', fontsize=8)
        axes[1, 1].grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        if SAVE_PLOTS_FLAG:
            model_file_name = f"{MODEL_TYPE}_analysis.png"
            output_file = os.path.join(DIRECTORY, model_file_name)
            plt.savefig(output_file, dpi=300)
            print(f"Figure saved to {output_file}")

        if show_plot:
            plt.show()

        # Return statistics
        return {
            "overall_accuracy": overall_accuracy,
            "accuracy_over_5dB": accuracy_over_5dB,
            "accuracy_per_snr": dict(zip(unique_snrs, accuracy_per_snr)),
            "peak_accuracy": peak_accuracy,
            "peak_snr": peak_snr,
        }

    def add_feature(name, func, *args):
        """Try to add a feature by checking the shape and ensuring it’s a scalar."""
        try:
            value = func(*args)
            # If value is an array, check if it is scalar (single value)
            if np.isscalar(value) or (isinstance(value, np.ndarray) and value.size == 1):
                feature_dict[name] = value.item() if isinstance(value, np.ndarray) else value
            else:
                print(f"Warning: Feature '{name}' has incorrect shape and was not added.")
        except Exception as e:
            print(f"Error computing feature '{name}': {e}")

    def extract_features(data):
        features = []
        labels = []
        snrs = []

        for key, signals in data.items():
            mod_type, snr = key
            for signal in signals:
                real_part, imag_part = signal[0], signal[1]
                complex_signal = real_part + 1j * imag_part

                # Reset feature dictionary for each signal
                global feature_dict
                feature_dict = {}

                # Add features with validation
                add_feature("Inst. Freq. Dev", instantaneous_frequency_deviation, complex_signal)
                add_feature("Spectral Entropy", spectral_entropy, real_part)
                add_feature("Envelope Mean", lambda x: envelope_mean_variance(x)[0], real_part)
                add_feature("Envelope Variance", lambda x: envelope_mean_variance(x)[1], real_part)
                add_feature("Spectral Flatness", spectral_flatness, real_part)
                add_feature("Spectral Peaks", lambda x: spectral_peaks_bandwidth(x)[0], real_part)
                add_feature("Bandwidth", lambda x: spectral_peaks_bandwidth(x)[1], real_part)
                add_feature("Zero Crossing Rate", zero_crossing_rate, real_part)
                add_feature("Amplitude Mean", lambda x: np.mean(compute_instantaneous_features(x)[0]), real_part)
                add_feature("Phase Variance", lambda x: np.var(compute_instantaneous_features(x)[1]), real_part)
                add_feature("Modulation Index", compute_modulation_index, real_part)
                add_feature("Spectral Sparsity", compute_spectral_asymmetry, real_part)
                add_feature("Envelope Ratio", lambda x: envelope_mean_variance(x)[0] / (envelope_mean_variance(x)[1] + 1e-10), real_part)
                add_feature("FFT Center Freq", lambda x: compute_fft_features(x)[0], real_part)
                add_feature("FFT Peak Power", lambda x: compute_fft_features(x)[1], real_part)
                add_feature("FFT Avg Power", lambda x: compute_fft_features(x)[2], real_part)
                add_feature("FFT Std Dev Power", lambda x: compute_fft_features(x)[3], real_part)
                add_feature("Kurtosis", compute_kurtosis, real_part)
                add_feature("Skewness", compute_skewness, real_part)
                add_feature("HOC-2", lambda x: compute_higher_order_cumulants(x, order=2), real_part)
                add_feature("HOC-3", lambda x: compute_higher_order_cumulants(x, order=3), real_part)
                add_feature("HOC-4", lambda x: compute_higher_order_cumulants(x, order=4), real_part)
                add_feature("Crest Factor", compute_crest_factor, real_part)
                add_feature("Spectral Entropy Value", compute_spectral_entropy, real_part)
                #add_feature("Autocorr Decay", compute_autocorrelation_decay, real_part)
                add_feature("RMS Instant Freq", compute_rms_of_instantaneous_frequency, real_part)
                add_feature("Entropy Instant Freq", compute_entropy_of_instantaneous_frequency, real_part)
                add_feature("Envelope Variance", compute_envelope_variance, real_part) #<< this one seems missing
                add_feature("PAPR", compute_papr, real_part)

                # Add SNR as a feature
                feature_dict["SNR"] = snr  # This has to be the last feature for the SNR Accuracy plot to work.

                # Append the feature values and label
                features.append(list(feature_dict.values()))
                labels.append(mod_type)

        return np.array(features), labels

    #Set a runtime timer
    start_time = time.time()

    ######## TIME STUFF IN HERE #########

    # Load the RML2016.10a_dict.pkl file with explicit encoding
    file_path = os.path.join(DIRECTORY, FILE_NAME)
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    elapsed_time = time.time() - start_time
    print(f"File Opened...{elapsed_time:.3f} seconds")

    # Feature extraction and scaling for all signals
    features, labels = extract_features(data)

    elapsed_time = time.time() - start_time
    print(f"Features Extracted...{elapsed_time:.3f} seconds")

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Encode labels for classification
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    elapsed_time = time.time() - start_time
    print(f"Features Scaled & Encoded...{elapsed_time:.3f} seconds")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.3, random_state=42)

    elapsed_time = time.time() - start_time
    print(f"Train and Test sets split...{elapsed_time:.3f} seconds")

    plot_all_analysis(X_test, y_test, label_encoder)

    ######## TIME STUFF IN HERE #########

    # Calcualte elapsed time
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.3f} seconds")