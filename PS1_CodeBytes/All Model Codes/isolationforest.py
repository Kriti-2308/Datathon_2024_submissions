import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

#df = pd.read_csv('Bank Account Fraud.csv') 
df = pd.read_csv('modified2.csv')
Y= df['fraud_bool'] 

df = df.drop(columns=['fraud_bool'])




categorical_columns = ['payment_type','employment_status', 'housing_status', 'source','device_os']
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

X = df.values
# Fit the Isolation Forest model
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X)

# Predict outliers
y_pred = clf.predict(X)
y_pred_binary = [0 if label == 1 else 1 for label in y_pred]
#print(y_pred_binary)


accuracy = accuracy_score(Y, y_pred_binary)
print("Accuracy:", accuracy)


# Add a new column to the dataframe indicating outliers
df['outlier'] = y_pred

# Visualize outliers
plt.scatter(df['foreign_request'], df['outlier'], c=df['outlier'], cmap='coolwarm')
#plt.scatter(c=df['outlier'], cmap='coolwarm')
plt.xlabel('income')
plt.ylabel('outlier')
plt.title('Anomaly Detection Results')
#plt.show()

# Print the number of outliers
num_outliers = (y_pred == -1).sum()
print("Number of outliers:", num_outliers)
auc_roc = roc_auc_score(Y, y_pred_binary)
precision = precision_score(Y, y_pred_binary)
recall = recall_score(Y, y_pred_binary)
f1 = f1_score(Y, y_pred_binary)

print("AUC-ROC Score:", auc_roc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

