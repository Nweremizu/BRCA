from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load the trained model and scaler
# Note: These will be created after we determine the best model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.joblib')

# Create models directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)

# Feature names from the breast cancer dataset
FEATURE_NAMES = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

@app.route('/')
def home():
    return render_template('index.html', feature_names=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from request
        features = []
        for feature in FEATURE_NAMES:
            value = float(request.form.get(feature, 0))
            features.append(value)
        
        # Convert to numpy array and reshape
        features = np.array(features).reshape(1, -1)
        
        # Load model and scaler
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'probability': float(probability[1]),  # Probability of malignant
            'interpretation': 'Malignant' if prediction == 1 else 'Benign'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 