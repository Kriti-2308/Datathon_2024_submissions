# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:20:05 2024

@author: avani
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score,accuracy_score
import tensorflow as tf

df =pd.read_csv("C:/Users/avani/Desktop/Datathon/modified.csv")

# Encode categorical columns
categorical_columns = ['payment_type','employment_status', 'housing_status','source','device_os']
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

X = df.iloc[:, 1:].values  # independent variable array
y = df.iloc[:, 0].values  # dependent variable vector
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#model = joblib.load('C:/Users/avani/Desktop/Datathon/Completed_NN_model.joblib')
model = tf.keras.models.load_model("C:/Users/avani/Desktop/Datathon/NN_model")
# Evaluate the model
y_pred_prob_nn = model.predict(X_test_scaled)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred_prob_nn]

# Calculate metrics
#accuracy = accuracy_score(y_test, y_pred_prob_nn)
auc_roc = roc_auc_score(y_test, y_pred_prob_nn)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

#print("Accuracy: ",accuracy)
print("AUC-ROC Score:", auc_roc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

y_pred_prob_nn = model.predict(X_test_scaled,verbose=0).round()
accuracy = accuracy_score(y_test, y_pred_prob_nn)
print("Accuracy: ",accuracy)