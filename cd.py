import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the training and test datasets
train_data = pd.read_csv('iith_foml_2023_train.csv')
test_data = pd.read_csv('iith_foml_2023_test.csv')

# Extract features and target variable from the training data
X = train_data.drop('Target Variable (Discrete)', axis=1)
Y = train_data['Target Variable (Discrete)']

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_val_standardized = scaler.transform(X_val)

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = imputer.fit_transform(X_train_standardized)
X_val_imputed = imputer.transform(X_val_standardized)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train_imputed, y_train)

# Make predictions on the validation set
y_pred = rf_classifier.predict(X_val_imputed)

# Evaluate the accuracy on the validation set
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

# # Standardize and impute the test data
# test_standardized = scaler.transform(test_data.drop('Id', axis=1))
# test_imputed = imputer.transform(test_standardized)

test_standardized = scaler.transform(test_data)
test_imputed = imputer.transform(test_standardized)

# Make predictions on the test data
test_predictions = rf_classifier.predict(test_imputed)

Id = []

for i in range(1, len(test_predictions)+1):
    Id.append(i)

output_df = pd.DataFrame({
    'Id': Id,
    'Category': test_predictions
})

# Specify the file path where you want to save the CSV file
output_file_path = 'iith_foml_2023_output.csv'

# Write the DataFrame to a CSV file
output_df.to_csv(output_file_path, index=False)

print(f"Results saved to '{output_file_path}'.")
    

# # Create a DataFrame with Id and predicted Category for the test predictions
# output_df = pd.DataFrame({'Id': test_data['Id'], 'Category': test_predictions})

# # Save the predictions to a CSV file
# output_df.to_csv('iith_foml_2023_output.csv', index=False)

# print("Prediction completed. Results saved to 'iith_foml_2023_output.csv'.")




