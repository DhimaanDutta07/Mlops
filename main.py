from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.train_pipeline import TrainPipeline
from src.utils.common import FileUtils
from src.config.config import ModelConfig

app = Flask(__name__, template_folder="frontend")

model_config = ModelConfig()

try:
    model = FileUtils.load_object(model_config.model_path)
except:
    model = TrainPipeline().run()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [
        float(request.form["area"]),
        float(request.form["bedrooms"]),
        float(request.form["bathrooms"]),
        float(request.form["stories"]),
        float(request.form["parking"])
    ]
    df = pd.DataFrame([data], columns=["area","bedrooms","bathrooms","stories","parking"])
    prediction = model.predict(df)[0]
    return render_template("index.html", result=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
