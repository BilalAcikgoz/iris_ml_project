# Python image
FROM python:3.11-slim

# Create Working Directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Open port
EXPOSE 8000

# Uvicorn ile başlat
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]