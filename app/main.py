from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.model import load_model, predict_species
import os

app = FastAPI()

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

# Static files
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Load model
model = load_model()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    features = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    result = predict_species(model, features)
    return templates.TemplateResponse("result.html", {"request": request, "prediction": result})

