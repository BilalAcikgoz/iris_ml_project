from sklearn.datasets import load_iris
import pandas as pd
import os

def export_to_csv():
    iris = load_iris(as_frame=True)
    df = iris.frame

    df["species"] = df["target"].map(lambda x: iris.target_names[x])
    df = df.drop(columns=["target"])

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/iris.csv", index=False)
    print("✅ Iris dataset exported to data/iris.csv")

if __name__ == "__main__":
    export_to_csv()