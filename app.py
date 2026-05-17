from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        values = [float(x) for x in request.form.values()]
        features = np.array([values])

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Fake Instagram Account "
        else:
            result = "Real Instagram Account "

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)