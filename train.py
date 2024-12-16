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

# Display processed data
print("Processed Data:\n", df.head())

# 3. Define Features and Target
# Features: Age, Gender, Condition, Medication, Dosage
X = df.drop(columns=['Dosage'])  # Assuming Dosage is target; replace with adherence column if available
y = df['Dosage']  # Replace with actual target variable

# Optional: Convert Dosage into categories for adherence (e.g., On-time, Missed, Delayed)
# y = pd.cut(y, bins=[0, 100, 500, 10000], labels=['On-time', 'Missed', 'Delayed'])

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 6. Save the Model
joblib.dump(rf_model, 'medication_adherence_model.pkl')
print("Model saved as 'medication_adherence_model.pkl'")

# 7. Flask API for Predictions
app = Flask(__name__)
model = joblib.load('medication_adherence_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])
    
    # Define features to match training data
    features = ['Age', 'Gender', 'Condition', 'Medication']  # Adjust based on your columns
    input_data = input_data[features]
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Add medication name to response
    medication_name = data.get('Medication_Name', 'Unknown')  # Optionally send Medication_Name in request
    return jsonify({
        'adherence_prediction': prediction[0],
        'medication_name': medication_name
    })

@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.json  # Expecting a list of medications
    predictions = []

    for record in data:
        input_data = pd.DataFrame([record])  # Convert to DataFrame
        features = ['Age', 'Gender', 'Condition', 'Medication']  # Match features
        input_data = input_data[features]
        adherence = model.predict(input_data)[0]

        predictions.append({
            'Medication_Name': record.get('Medication_Name', 'Unknown'),
            'Adherence': adherence
        })
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
