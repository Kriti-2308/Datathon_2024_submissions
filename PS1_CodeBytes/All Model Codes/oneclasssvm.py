import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Load the dataset
df = pd.read_csv('Bank Account Fraud.csv')  # Replace 'your_dataset.csv' with the path to your dataset
Y= df['fraud_bool'] 

df = df.drop(columns=['fraud_bool'])




categorical_columns = ['payment_type','employment_status', 'housing_status', 'source','device_os']
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(df.values)

# Train the One-Class SVM model
clf = OneClassSVM(nu=0.1)  # Adjust the 'nu' parameter as needed
clf.fit(X)

# Predict outliers
y_pred = clf.predict(X)

# Convert outlier predictions to binary (1 for outliers, -1 for inliers)
y_pred_binary = [1 if pred == -1 else 0 for pred in y_pred]

# Calculate accuracy (if labels are available)
# Replace 'labels' with the actual labels if available
labels = Y  # Replace 'label_column' with the name of the column containing labels
accuracy = accuracy_score(labels, y_pred)
precision = precision_score(labels, y_pred)
recall = recall_score(labels, y_pred)
f1 = f1_score(labels, y_pred)
auc_roc = roc_auc_score(labels, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC-ROC Score:", auc_roc)


# Visualize outliers (optional)
# import matplotlib.pyplot as plt
# plt.scatter(df['feature1'], df['feature2'], c=y_pred_binary, cmap='coolwarm')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Anomaly Detection Results')
# plt.show()
