from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.components.pipeline.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application  

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

## Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        # Corrected 'request.from' → 'request.form'
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race"),
            parental_level_of_education=request.form.get("education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("prep"),
            reading_score=float(request.form.get("reading")),
            writing_score=float(request.form.get("writing"))
        )

        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame:\n", pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=results[0])
if __name__ == "__main__":
    app.run(host="0.0.0.0")