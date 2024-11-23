import time
import os
import sys
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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

    #Set a runtime timer for the training only
    start_time = time.time()

    # Load the RML2016.10a_dict.pkl file with explicit encoding
    file_path = os.path.join(DIRECTORY, FILE_NAME)
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # Feature extraction for all signals
    features, labels = extract_features(data)

    # Encode labels for classification
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # ######## DATA EDA ###########################################
    # print(feature_dict)
    # df = pd.DataFrame(features)
    # print("====================================")
    # print("Basic Dataframe info")
    # print(df.info())
    # print("====================================")
    # print("Number of Empty values in each column:")
    # print(df.isnull().sum().sort_values(ascending=False))
    # print("====================================")
    # ######## DATA EDA ###########################################

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.3, random_state=42)

    # Train a single classifier on the entire dataset for multi-class classification
    # RND_FOREST, SVC_LINEAR, SVC_POLY, SVC_RBF,
    match MODEL_TYPE:
        case "RND_FOREST":
            print("Using Random Forest")
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
        case "SVC_LINEAR":
            print("Using SVC with linear kernel")
            clf = SVC(kernel='linear', random_state=42)
        case "SVC_POLY":
            print("Using SVC with polynomial kernel")
            clf = SVC(kernel='poly', degree=2, gamma='auto', coef0=1, C=5)
        case "SVC_RBF":
            print("Using SVC with radial basis function kernel")
            clf = SVC(kernel='rbf', gamma=0.001, C=100000.0)
        case _:
            print("Unknown Model Type. Exiting!")
            sys.exit()

    clf.fit(X_train, y_train)

    #Calcualte elapsed time for training only
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the model: {elapsed_time:.3f} seconds")

    # Save the trained model to a file with the dynamic name
    model_file_name = f"model_{MODEL_TYPE}.pkl"
    model_path = os.path.join(DIRECTORY, model_file_name)
    joblib.dump(clf, model_path)

    # Evaluate accuracy for each SNR level
    unique_snrs = sorted(set(X_test[:, -1]))  # Get unique SNR levels from test set
    accuracy_per_snr = []

    for snr in unique_snrs:
        # Select samples with the current SNR
        snr_indices = np.where(X_test[:, -1] == snr)
        X_snr = X_test[snr_indices]
        y_snr = y_test[snr_indices]

        # Predict and calculate accuracy
        y_pred = clf.predict(X_snr)
        accuracy = accuracy_score(y_snr, y_pred)
        accuracy_per_snr.append(accuracy * 100)  # Convert to percentage

        print(f"SNR: {snr} dB, Accuracy: {accuracy * 100:.2f}%")

    # Plot Recognition Accuracy vs. SNR
    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, accuracy_per_snr, 'b-o', label='Recognition Accuracy')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Recognition Accuracy (%)")
    plt.title("Recognition Accuracy vs. SNR for Modulation Classification")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100)
    if SAVE_PLOTS_FLAG:
        model_file_name = f"{MODEL_TYPE}_Accuracy_vs_SNR.png"
        save_path = os.path.join(DIRECTORY, model_file_name)
        plt.savefig(save_path)
    plt.show(block=False)

    # Feature importance for the classifier
    if MODEL_TYPE == "RND_FOREST":
        feature_names = list(feature_dict.keys())
        importances = clf.feature_importances_
        plt.figure(figsize=(12, 8))
        plt.barh(feature_names, importances, color='skyblue')
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance for Modulation Classification")
        if SAVE_PLOTS_FLAG:
            model_file_name = f"{MODEL_TYPE}_FeatureImportance.png"
            save_path = os.path.join(DIRECTORY, model_file_name)
            plt.savefig(save_path)
        plt.show(block=False)

    # Confusion matrix for overall test set
    y_pred_test = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Multi-Class Modulation Classification")
    if SAVE_PLOTS_FLAG:
        model_file_name = f"{MODEL_TYPE}_ConfusionMatrix.png"
        save_path = os.path.join(DIRECTORY, model_file_name)
        plt.savefig(save_path)
    plt.show(block=False)

    # Print Classification Report
    print("Classification Report for Modulation Types:")
    print("Train Result:n================================================")
    # Confusion matrix for overall test set
    y_pred_train = clf.predict(X_train)
    print(classification_report(y_train, y_pred_train, target_names=label_encoder.classes_))

    print("Test Result:n================================================")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    # # Filter test samples with SNR > 5 dB
    # snr_above_5_indices = np.where(X_test[:, -1] > 5)  # Select indices where SNR > 5
    # X_test_snr_above_5 = X_test[snr_above_5_indices]
    # y_test_snr_above_5 = y_test[snr_above_5_indices]
    #
    # # Predict on this subset of data
    # y_pred_snr_above_5 = clf.predict(X_test_snr_above_5)
    #
    # # Plot confusion matrix for SNR > 5 dB
    # conf_matrix_snr_above_5 = confusion_matrix(y_test_snr_above_5, y_pred_snr_above_5)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(conf_matrix_snr_above_5, annot=True, fmt="d", cmap="Blues",
    #             xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.title("Confusion Matrix for Modulation Classification (SNR > 5 dB)")
    # plt.show()
    #
    # # Print Classification Report for SNR > 5 dB
    # print("Classification Report for Modulation Types (SNR > 5 dB):")
    # print(classification_report(y_test_snr_above_5, y_pred_snr_above_5, target_names=label_encoder.classes_))