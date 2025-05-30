import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/iris.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")