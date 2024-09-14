import pandas as pd
import numpy as np
import joblib

model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

def detect_fraud(new_data):
    required_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                        'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    new_data = new_data[required_columns]
    
    new_data_imputed = imputer.transform(new_data)
    
    new_data_scaled = scaler.transform(new_data_imputed)
    
    probabilities = model.predict_proba(new_data_scaled)
    
    threshold = 0.5
    fraud_predictions = (probabilities[:, 1] > threshold).astype(int)
    
    return fraud_predictions

if __name__ == "__main__":
    new_transactions = pd.DataFrame({
        'Time': [0.0, 1.0],
        'V1': np.random.randn(2),
        'V2': np.random.randn(2),
        'V3': np.random.randn(2),
        'V4': np.random.randn(2),
        'V5': np.random.randn(2),
        'V6': np.random.randn(2),
        'V7': np.random.randn(2),
        'V8': np.random.randn(2),
        'V9': np.random.randn(2),
        'V10': np.random.randn(2),
        'V11': np.random.randn(2),
        'V12': np.random.randn(2),
        'V13': np.random.randn(2),
        'V14': np.random.randn(2),
        'V15': np.random.randn(2),
        'V16': np.random.randn(2),
        'V17': np.random.randn(2),
        'V18': np.random.randn(2),
        'V19': np.random.randn(2),
        'V20': np.random.randn(2),
        'V21': np.random.randn(2),
        'V22': np.random.randn(2),
        'V23': np.random.randn(2),
        'V24': np.random.randn(2),
        'V25': np.random.randn(2),
        'V26': np.random.randn(2),
        'V27': np.random.randn(2),
        'V28': np.random.randn(2),
        'Amount': [10.0, 20.0]
    })
    
    fraud_predictions = detect_fraud(new_transactions)
    print("Fraud Predictions:", fraud_predictions)
    
    if any(fraud_predictions):
        print("Alert: Fraudulent transactions detected!")
