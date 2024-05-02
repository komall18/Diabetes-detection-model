import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting features from form data
    input_data = {
        'remainder__age': float(request.form['remainder__age']),
        'remainder__hypertension': float(request.form['remainder__hypertension']),
        'remainder__bmi': float(request.form['remainder__bmi']),
        'remainder__HbA1c_level': float(request.form['remainder__HbA1c_level']),
        'remainder__blood_glucose_level': float(request.form['remainder__blood_glucose_level'])
    }

    # Making prediction
    selected_features = ['remainder__age', 'remainder__hypertension', 'remainder__bmi', 'remainder__HbA1c_level', 'remainder__blood_glucose_level']
    f = np.array([input_data[feature] for feature in selected_features]).reshape(1, -1)
    prediction = model.predict(f)

    # Formatting the prediction result
    prediction_text = "Prediction: {} (indicating {})".format(prediction[0], "diabetes" if prediction[0] == 1 else "no diabetes")

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
