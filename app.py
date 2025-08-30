from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model at startup
model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Expect data to contain "total_value" feature for prediction
    try:
        total_value = float(data['total_value'])
    except (KeyError, ValueError):
        return jsonify({"error": "Invalid input"}), 400
    pred = model.predict(np.array([[total_value]]))
    return jsonify({"predicted_daily_active_users": float(pred[0])})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
