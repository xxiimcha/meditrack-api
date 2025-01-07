import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from flask import Flask, request, jsonify
import joblib

# 1. Load Dataset
df = pd.read_csv('dataset.csv')  # Replace with your dataset file path
print("Original Data:\n", df.head())

# 2. Data Preprocessing
# Drop ID column if not needed
df = df.drop(columns=['ID'], errors='ignore')

# Encode Categorical Columns
le = LabelEncoder()
for col in ['Gender', 'Condition', 'Medication']:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Check for missing values
df = df.dropna()

# Convert Dosage to numeric values by extracting the lower bound of the range
def preprocess_dosage(dosage):
    if isinstance(dosage, str):
        # Extract the lower bound from the range (e.g., '5-10 mg/day' → 5)
        numeric_part = dosage.split('-')[0].strip()  # Get the part before '-'
        numeric_part = ''.join(filter(str.isdigit, numeric_part))  # Extract only digits
        if numeric_part.isdigit():
            return int(numeric_part)
        else:
            print(f"Warning: Could not process dosage value '{dosage}'")  # Log unprocessed values
            return 0
    elif isinstance(dosage, (int, float)):
        return int(dosage)  # Handle numeric values
    else:
        print(f"Warning: Unknown format for dosage value '{dosage}'")  # Log unknown formats
        return 0  # Default to 0 for unprocessed values

df['Dosage'] = df['Dosage'].apply(preprocess_dosage)

# Check if Dosage is being classified correctly
print("Processed Dosage Column:\n", df['Dosage'].value_counts())

# Convert Dosage to Binary Classification
# 0 = missed, 1 = taken
df['Dosage'] = df['Dosage'].apply(lambda x: 1 if x > 0 else 0)

# 3. Define Features and Target
# Features: Age, Gender, Condition, Medication
features = df.drop(columns=['Dosage'])  # Features
target_adherence = df['Dosage']  # Binary target (0: missed, 1: taken)

# 4. Train-Test Split
features_train, features_test, target_train, target_test = train_test_split(
    features, target_adherence, test_size=0.2, random_state=42
)

# 5. Train Random Forest Model
adherence_model = RandomForestClassifier(n_estimators=100, random_state=42)
adherence_model.fit(features_train, target_train)

# 6. Save the Model
joblib.dump(adherence_model, 'medication_adherence_model.pkl')
print("Model saved as 'medication_adherence_model.pkl'")

# 7. Flask API for Predictions
app = Flask(__name__)
model = joblib.load('medication_adherence_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    
    # Define features to match training data
    feature_columns = ['Age', 'Gender', 'Condition', 'Medication']  # Adjust based on your columns
    input_data = input_data[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Add medication name to response
    medication_name = data.get('Medication_Name', 'Unknown')  # Optionally send Medication_Name in request
    return jsonify({
        'adherence_prediction': int(prediction[0]),  # Convert prediction to int for JSON serialization
        'medication_name': medication_name
    })

@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.json  # Expecting a list of medications
    predictions = []

    for record in data:
        input_data = pd.DataFrame([record])  # Convert to DataFrame
        feature_columns = ['Age', 'Gender', 'Condition', 'Medication']  # Match features
        input_data = input_data[feature_columns]
        adherence = model.predict(input_data)[0]

        predictions.append({
            'Medication_Name': record.get('Medication_Name', 'Unknown'),
            'Adherence': int(adherence)  # Convert prediction to int
        })
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
