from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
ENCODER_PATH = os.path.join(MODEL_DIR, 'encoder.joblib')

model = None
scaler = None
encoder = None

def load_artifacts():
    global model, scaler, encoder
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        return True
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return False

@app.route('/')
def index():
    """Landing / Home page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction form and result page."""
    if model is None:
        load_artifacts()

    if request.method == 'GET':
        return render_template('predict.html', prediction=None)

    # POST — process form
    if model is None:
        return render_template('predict.html',
                               error="Model artifacts not found. Please run 'train.py' first.")
    try:
        form_data = request.form.to_dict()
        gender = 1 if form_data['Gender'] == 'Male' else 0

        input_data = [
            gender,
            float(form_data['Age']),
            float(form_data['Hemoglobin']),
            float(form_data['RBC']),
            float(form_data['WBC']),
            float(form_data['AST']),
            float(form_data['ALT']),
            float(form_data['Cholestrol']),
            float(form_data['Spirometry']),
            float(form_data['Creatinine']),
            float(form_data['Glucose']),
            float(form_data['Lipase']),
            float(form_data['Troponin'])
        ]

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        prediction_encoded = model.predict(input_scaled)[0]
        prediction_text = encoder.inverse_transform([prediction_encoded])[0]

        probabilities = model.predict_proba(input_scaled)[0]
        confidence = np.max(probabilities) * 100

        return render_template('predict.html',
                               prediction=prediction_text,
                               confidence=round(confidence, 2),
                               form_data=form_data)

    except Exception as e:
        return render_template('predict.html',
                               error=f"An error occurred during prediction: {str(e)}")

@app.route('/about')
def about():
    """About page describing the model and methodology."""
    return render_template('about.html')

if __name__ == '__main__':
    load_artifacts()
    app.run(debug=True, port=5000)
