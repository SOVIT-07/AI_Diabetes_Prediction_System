from flask import Flask, render_template, request

from src.predict import predict_diabetes


app = Flask(__name__)


FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]


@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        try:

            input_data = {}

            for feature in FEATURES:

                value = request.form.get(
                    feature
                )

                if value is None or value == "":
                    raise ValueError(
                        f"Missing value: {feature}"
                    )

                input_data[feature] = float(
                    value
                )

            result = predict_diabetes(
                input_data
            )

        except Exception as exc:

            error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )