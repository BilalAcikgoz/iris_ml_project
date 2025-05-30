import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from app.config import DATA_PATH, MODEL_PATH

def train():
    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df.drop("species", axis=1)
    y = df["species"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✅ Model trained with accuracy: {acc:.4f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train()