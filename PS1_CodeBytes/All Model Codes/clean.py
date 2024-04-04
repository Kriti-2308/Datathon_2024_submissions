import pandas as pd

# Load the dataset
dataset =pd.read_csv("C:/Users/avani/Desktop/Datathon/Bank Account Fraud.csv")

# List of numeric columns with negative values as missing
negative_numeric_features = ['prev_address_months_count', 'current_address_months_count',
                             'intended_balcon_amount', 'velocity_6h', 'velocity_24h',
                             'velocity_4w', 'session_length_in_minutes', 'bank_months_count',]

# Replace negative values with NaN
for feature in negative_numeric_features:
    dataset.loc[dataset[feature] < 0, feature] = pd.NA

# List of categorical columns
categorical_features = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']

# Fill NaNs with mode for categorical features
for feature in categorical_features:
    dataset[feature].fillna(dataset[feature].mode()[0], inplace=True)

# Fill remaining missing values with mode for all columns
dataset.fillna(dataset.mode().iloc[0], inplace=True)

# Save the modified dataset to another CSV file
dataset.to_csv("modified.csv", index=False)
