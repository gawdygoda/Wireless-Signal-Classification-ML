import time
import os
import sys
from feature_extraction import extract_features
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import numpy as np
from matplotlib.colors import Normalize
import joblib
from joblib import parallel_backend

DIRECTORY = "./data"
FILE_NAME = "RML2016.10a_dict.pkl"
MODEL_TYPE = "SVC_POLY_C" #SVC_LINEAR_C, SVC_POLY_C, SVC_RBF_C
SAVE_PLOTS_FLAG = 1

# n_jobs=-2 means run on all CPUs - 1 (leave one for me to surf the web!)
with parallel_backend('threading', n_jobs=-2):

    #Class for colorizing RBF plots
    class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    #Set a runtime timer for the training only
    start_time = time.time()

    # Input file to pass to the function
    file_path = os.path.join(DIRECTORY, FILE_NAME)

    features_df = extract_features(file_path)
    #print (features_df.shape )
    #count = (features_df['signal_type'] == 'BPSK').sum()
    #print(count)

    # Create new dataframe for target variable or label column for supervised learning
    y = pd.DataFrame(features_df['signal_type'])

    # Label encoding the target variable
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)

    # # One-hot encoding the target variable
    # encoder = OneHotEncoder(sparse_output=False)
    # y_encoded = encoder.fit_transform(features_df[['signal_type']])
    # y = y_encoded.argmax(axis=1)  # Convert back to 1D array of class labels

    training_features = ["snr", "magnitude_mean", "magnitude_std", "magnitude_skew", "magnitude_kurtosis",
                    "phase_mean", "phase_std", "phase_skew", "phase_kurtosis", "spectral_entropy", "peak_frequency",
                    "average_power"]

    # Create new dataframe for features variables or training columns for supervised learning
    feature_transform = features_df[training_features]
    X = pd.DataFrame(columns=training_features, data=feature_transform, index=features_df.index)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #Split the dataset for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.3, random_state=42)

    # ###########################   SVC RBF Parameter Grid Search + Model - START
    #
    # # This code will do
    # # num C_range x num gama_range x n_splits iterations
    # # 5*5*1 iterations = 25 iterations
    # # from testing, each iteration takes ?? min to train on M1 Mac Pro (Iterations take longer as the numbers get bigger)
    # C_range = np.logspace(-2, 5, 5)
    # gamma_range = np.logspace(-9, 3, 5)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    #
    # print (param_grid)
    #
    # cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    #
    # grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)#, n_jobs=-2)
    # grid.fit(X_train, y_train)
    #
    # print(
    #     "The best parameters are %s with a score of %0.2f"
    #     % (grid.best_params_, grid.best_score_)
    # )
    #
    # ###########################   SVC RBF Parameter Grid Search - END


    ###########################   SVC Model - START

    # Train a single classifier on the entire dataset for multi-class classification
    # SVC_LINEAR_C, SVC_POLY_C, SVC_RBF_C,
    match MODEL_TYPE:
        case "SVC_LINEAR_C":
            print("Using SVC with linear kernel")
            svm_model = SVC(kernel='linear', random_state=42)
        case "SVC_POLY_C":
            print("Using SVC with polynomial kernel")
            svm_model = SVC(kernel='poly', degree=2, gamma='auto', coef0=1, C=5)
        case "SVC_RBF_C":
            print("Using SVC with radial basis function kernel")
            svm_model = SVC(kernel='rbf', gamma=0.001, C=100000.0)
        case _:
            print("Unknown Model Type. Exiting!")
            sys.exit()

    ###########################   SVC Model - END

    # Train the model
    svm_model.fit(X_train, y_train)

    #Calcualte elapsed time for training only
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the model: {elapsed_time:.3f} seconds")

    # Save the trained model to a file with the dynamic name
    model_file_name = f"model_{MODEL_TYPE}.pkl"
    model_path = os.path.join(DIRECTORY, model_file_name)
    joblib.dump(svm_model, model_path)

    ###########################   SVC Model PREDICTION - START

    # Predictions on the  dataset
    y_pred_test = svm_model.predict(X_test)
    y_pred_train = svm_model.predict(X_train)

    ###########################   SVC Model PREDICTION - END


    # # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred_test)
    # print(f"Accuracy: {accuracy:.2f}")

    # Print Classification Report
    print("Classification Report for Modulation Types:")
    print("Train Result:n================================================")
    # Confusion matrix for overall test set
    print(classification_report(y_train, y_pred_train, target_names=label_encoder.classes_))
    print("Test Result:n================================================")
    print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_))

    ###########################   Confusion matrix - START
    confusionMatrix = confusion_matrix(y_test, y_pred_test)
    print("Confusion Matrix:")
    print(confusionMatrix)

    plt.figure(figsize=(11, 11))
    sns.heatmap(confusionMatrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Signal Type Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    if SAVE_PLOTS_FLAG:
        model_file_name = f"{MODEL_TYPE}_ConfusionMatrix.png"
        save_path = os.path.join(DIRECTORY, model_file_name)
        plt.savefig(save_path)
    plt.show(block=False)

    ###########################   Confusion matrix - END

    ###########################   Accuracy vs SNR - START

    # Evaluate accuracy for each SNR level
    #X_test = scaler.inverse_transform(X_test)
    unique_snrs = sorted(set(X_test[:, 0]))  # Get unique SNR levels from test set
    accuracy_per_snr = []

    for snr in unique_snrs:
        # Select samples with the current SNR
        snr_indices = np.where(X_test[:, 0] == snr)
        X_snr = X_test[snr_indices]
        y_snr = y_test[snr_indices]

        # Predict and calculate accuracy
        y_pred = svm_model.predict(X_snr)
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

###########################   Accuracy vs SNR - END