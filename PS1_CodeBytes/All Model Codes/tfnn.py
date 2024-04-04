# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:47:02 2024

@author: avani
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import tensorflow as tf
# Load the dataset
df =pd.read_csv("C:/Users/avani/Desktop/Datathon/modified.csv")

# Encode categorical columns
categorical_columns = ['payment_type','employment_status', 'housing_status','source','device_os']
label_encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# Split the dataset into features (X) and target variable (y)
X = df.iloc[:, 1:].values  # independent variable array
y = df.iloc[:, 0].values  # dependent variable vector

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
#model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Save the model
model.save("NN_model_20")


# Evaluate the model
y_pred_prob_nn = model.predict(X_test_scaled)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred_prob_nn]

# Calculate metrics
auc_roc = roc_auc_score(y_test, y_pred_prob_nn)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("AUC-ROC Score:", auc_roc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)