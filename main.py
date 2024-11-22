import time
from feature_extraction import extract_features
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
import numpy as np
from matplotlib.colors import Normalize
from joblib import parallel_backend

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
    data_file = "./data/RML2016.10a_dict.pkl"

    features_df = extract_features(data_file)
    #print (features_df.shape )
    #count = (features_df['signal_type'] == 'BPSK').sum()
    #print(count)

    # Create new dataframe for target variable or label column for supervised learning
    y = pd.DataFrame(features_df['signal_type'])

    # # Label encoding the target variable
    # encoder = LabelEncoder()
    # encoded_labels = encoder.fit_transform(y)

    # # One-hot encoding the target variable
    # encoder = OneHotEncoder(sparse_output=False)
    # y_encoded = encoder.fit_transform(features_df[['signal_type']])
    # y = y_encoded.argmax(axis=1)  # Convert back to 1D array of class labels

    training_features = ["magnitude_mean", "magnitude_std", "magnitude_skew", "magnitude_kurtosis",
                    "phase_mean", "phase_std", "phase_skew", "phase_kurtosis", "spectral_entropy", "peak_frequency",
                    "average_power"]

    # Create new dataframe for features variables or training columns for supervised learning
    feature_transform = features_df[training_features]
    X = pd.DataFrame(columns=training_features, data=feature_transform, index=features_df.index)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #Split the dataset for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # ###########################   SVC Linear Model - START
    #
    # # Create an SVM classifier
    # svm_model = SVC(kernel='linear', random_state=42)
    # ###########################   SVC Linear Model - END


    ###########################   SVC RBF Model - START

    # Create an SVM classifier
    svm_model = SVC(kernel='rbf', gamma=0.001, C=100000.0)
    ###########################   SVC RBF Model - END

    # Train the model
    svm_model.fit(X_train, y_train)

    #Calcualte elapsed time for training only
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the model: {elapsed_time:.3f} seconds")

    ###########################   SVC Linear Model PREDICTION - START

    # Predictions on the test set
    y_pred = svm_model.predict(X_test)

    ###########################   SVC Linear Model PREDICTION - END


    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    ###########################   Confusion matrix - START
    confusionMatrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusionMatrix)

    plt.figure(figsize=(11, 11))
    sns.heatmap(confusionMatrix, annot=True, annot_kws={"fontsize": 12}, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, linecolor="black")
    plt.title("Signal Type Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("Actual", fontsize=14)
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], labels=['AM-SSB', 'WBFM', 'BPSK', 'CPFSK', 'GFSK', 'AM-DSB', 'QAM16', 'QAM64', 'QPSK', '8PSK', 'PAM4'], fontsize=10)
    plt.yticks(ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], labels=['AM-SSB', 'WBFM', 'BPSK', 'CPFSK', 'GFSK', 'AM-DSB', 'QAM16', 'QAM64', 'QPSK', '8PSK', 'PAM4'], fontsize=10)
    plt.show()

    ###########################   Confusion matrix - END
