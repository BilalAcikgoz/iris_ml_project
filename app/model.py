import pickle
from app.schema import IrisFeatures

def load_model(model_path: str = "models/model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_species(model, features: IrisFeatures):
    data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = model.predict(data)[0]
    return {"predicted_class": prediction}