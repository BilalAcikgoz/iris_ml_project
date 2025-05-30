# 🌸 Iris Species Prediction API

A lightweight, production-ready machine learning API built with FastAPI. This project predicts the species of iris flowers (Setosa, Versicolor, Virginica) based on four features using a trained RandomForest model.

## 🚀 Features

- 🧠 Trained RandomForestClassifier (100% test accuracy)
- ⚡ Fast and asynchronous API with FastAPI
- 🔍 Input validation using Pydantic
- 🌐 Simple HTML frontend for user input
- 🐳 Docker support for deployment
- 🧪 Unit tests with Pytest

---

## 🧰 Technologies Used

- Python 3.11
- FastAPI
- Scikit-learn
- Pydantic
- Jinja2
- Docker
- Pytest

---

## ⚙️ Getting Started

### 1. Set up virtual environment and install dependencies

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv run ml/train.py
uvicorn app.main:app --reload
pytest tests/

📌 Notes

- Based on the classic Iris dataset from sklearn.datasets
- Designed to be easily extended with MLflow, database logging, or CI/CD
- Ideal for beginner-to-intermediate MLOps projects

👨‍💻 Author
Bilal Açıkgöz
AI Developer & AI Enthusiast