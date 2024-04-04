import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import lightgbm as lgb

# Load the dataset
#df = pd.read_csv('Bank Account Fraud.csv') 
df = pd.read_csv('modified.csv')

# Encode categorical columns
categorical_columns = ['payment_type','employment_status', 'housing_status','source','device_os']
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Split the dataset into features (X) and target variable (y)
X = df.iloc[:, 1:].values  # independent variable array
y = df.iloc[:, 0].values  # dependent variable vector

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Set parameters for LightGBM
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_error',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the LightGBM model
num_round = 100
bst = lgb.train(params, train_data, num_round)

# Predict on test data
y_pred_lgbm = bst.predict(X_test)

# Convert probabilities to binary predictions
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred_lgbm]

# Calculate accuracy
accuracy_lgbm = accuracy_score(y_test, y_pred_binary)
print("LightGBM Accuracy:", accuracy_lgbm)

auc_roc = roc_auc_score(y_test, y_pred_binary)
print("AUC-ROC Score:", auc_roc)

# Calculate precision
precision = precision_score(y_test, y_pred_binary)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred_binary)
print("Recall:", recall)

# Calculate F1 score
f1 = f1_score(y_test, y_pred_binary)
print("F1 Score:", f1)