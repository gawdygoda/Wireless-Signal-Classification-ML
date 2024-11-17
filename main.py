from feature_extraction import extract_features
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import pandas as pd
# import seaborn as sns



# Input file to pass to the function
data_file = "./data/RML2016.10a_dict.pkl"

features_df = extract_features(data_file)
print (features_df.shape )

count = (features_df['signal_type'] == 'BPSK').sum()

print(count)


# Create new dataframe for target variable or label column for supervised learning
y = pd.DataFrame(features_df['signal_type'])

training_features = ["magnitude_mean", "magnitude_std", "magnitude_skew", "magnitude_kurtosis",
                "phase_mean", "phase_std", "phase_skew", "phase_kurtosis", "spectral_entropy", "peak_frequency",
                "average_power"]

# Create new dataframe for features variables or training columns for supervised learning
feature_transform = features_df[training_features]
X = pd.DataFrame(columns=training_features, data=feature_transform, index=features_df.index)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for SVM performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier (use linear kernel in this example)
svm_model = SVC(kernel='linear', random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

auc_roc = roc_auc_score(y_test, y_pred)
print("AUC-ROC:", auc_roc)

# Confusion matrix
confusionMatrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusionMatrix)

# Plotting confusion matrix
plt.figure(figsize=(5, 5))
sns.heatmap(confusionMatrix, annot=True, annot_kws={"fontsize": 12}, fmt="d", cmap="Blues", cbar=False, linewidths=0.5, linecolor="black")
plt.title("Signal Type Confusion Matrix", fontsize=14)
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("Actual", fontsize=14)
plt.xticks(ticks=[0.5, 1.5], labels=["BPSK","QPSK","QAM16","WBFM","GFSK"], fontsize=10)
plt.yticks(ticks=[0.5, 1.5], labels=["BPSK","QPSK","QAM16","WBFM","GFSK"], fontsize=10)
plt.show()