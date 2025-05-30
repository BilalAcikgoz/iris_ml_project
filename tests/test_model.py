import os
import pickle
from app.config import MODEL_PATH
from app.schema import IrisFeatures

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model file does not exist at {MODEL_PATH}"

def test_model_prediction():
    # Modeli yükle
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Örnek veri
    sample = IrisFeatures(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2
    )

    # Tahmin yap
    prediction = model.predict([[sample.sepal_length, sample.sepal_width, sample.petal_length, sample.petal_width]])

    # Sonucun boş olmaması yeterli
    assert prediction[0] in ["setosa", "versicolor", "virginica"]
