import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('Bank Account Fraud.csv') 
Y = df['fraud_bool'] 
df = df.drop(columns=['fraud_bool'])

# Encoding categorical columns
categorical_columns = ['payment_type','employment_status', 'housing_status', 'source','device_os']
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Fit DBSCAN model
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust epsilon and min_samples as needed
y_pred = dbscan.fit_predict(X)

# Convert predicted labels to 0 for normal points and 1 for anomalies
y_pred_binary = [1 if label == -1 else 0 for label in y_pred]

# Calculate accuracy
accuracy = accuracy_score(Y, y_pred_binary)
print("Accuracy:", accuracy)
