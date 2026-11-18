
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained ML model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

# Get input feature names from trained model
features = list(model.feature_names_in_)


@app.route("/")
def home():
    return render_template("index.html", features=features)


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get student input values
        input_data = {}

        for feature in features:
            value = float(request.form[feature])
            input_data[feature] = value

        # Convert input into DataFrame
        student_data = pd.DataFrame([input_data], columns=features)

        # Predict final marks
        prediction = model.predict(student_data)[0]

        prediction = round(float(prediction), 2)

        return render_template(
            "index.html",
            features=features,
            prediction=prediction
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            features=features,
            error="Please enter valid numerical values."
        )


if __name__ == "__main__":
    app.run(debug=True)
