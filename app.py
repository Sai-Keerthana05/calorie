import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

# Initialize the Flask app
app = Flask(__name__)

# Load the model and scaler
scaler = joblib.load('scaler.pkl')  # Adjust the path if needed
model = joblib.load('calorie_model.pkl')  # Adjust the path if needed

# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Route to predict calorie burn based on user input
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = float(request.form['age'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    gender = 1 if request.form['gender'] == 'female' else 0  # Convert gender to numeric
    body_temp = float(request.form['body_temp'])
    duration = float(request.form['duration'])
    heart_rate = float(request.form['heart_rate'])

    # Create a feature array
    features = np.array([[age, height, weight, gender, body_temp, duration, heart_rate]])

    # Scale the features
    features_scaled = scaler.transform(features)

    # Make the prediction
    prediction = model.predict(features_scaled)

    # Render the result
    return render_template('index.html', prediction_text=f"Estimated Calories Burnt: {prediction[0]:.2f} kcal")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
