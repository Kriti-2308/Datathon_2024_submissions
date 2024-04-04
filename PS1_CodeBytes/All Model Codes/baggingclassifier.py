import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset
df = pd.read_csv("C:/Users/avani/Desktop/Datathon/Bank Account Fraud.csv")
#df = pd.read_csv("C:/Users/avani/Desktop/Datathon/modified.csv")

# Encode categorical columns
categorical_columns = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Split the dataset into features (X) and target variable (y)
X = df.iloc[:, 1:].values  # independent variable array
y = df.iloc[:, 0].values  # dependent variable vector

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create base classifier (Decision Tree)
base_classifier = DecisionTreeClassifier()

# Create Bagging Classifier
bagging_clf = BaggingClassifier(base_classifier, n_estimators=10, random_state=0)

# Train the Bagging Classifier
bagging_clf.fit(X_train, y_train)

# Predict probabilities on test data for calculating AUC-ROC score
y_pred_prob = bagging_clf.predict_proba(X_test)[:, 1]

# Predict on test data
y_pred = bagging_clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_prob)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC-ROC Score:", auc_roc)
