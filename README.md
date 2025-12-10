🚕 NYC Taxi Trip Duration – End-to-End MLOps Project

This project implements a production-ready MLOps pipeline for predicting NYC Taxi Trip Duration, covering data processing, feature engineering, model training, versioning, containerization, CI/CD, and deployment using Docker, Kubernetes, and Seldon Core.

📌 Problem Statement

Given pickup and drop-off locations along with time information, predict the expected taxi trip duration in seconds.

This is a regression problem based on the well-known NYC Taxi Trip Duration dataset.

🧠 Key Highlights

✅ Modular data pipeline (Cookiecutter style)

✅ Advanced feature engineering (distance, bearing, time features)

✅ XGBoost model for regression

✅ DVC for data & model versioning

✅ FastAPI for model serving

✅ Docker containerization

✅ Kubernetes deployment

✅ Seldon Core for ML serving (optional)

✅ GitHub Actions CI pipeline

🗂️ Project Structure
trip-duration-mlops/
├── data/
│   ├── raw/                # Raw NYC taxi CSV data
│   └── processed/          # Feature-engineered data (DVC tracked)
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── feature_definitions.py
│   └── models/
│       └── train_model.py
├── service.py              # FastAPI inference service
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Training parameters (optional)
├── Dockerfile
├── deployment.yaml         # Kubernetes deployment
├── seldon_deployment.yaml  # Seldon Core deployment
├── requirements.txt
├── .gitignore
└── .github/workflows/ci.yml

⚙️ Tech Stack

Language: Python

ML: Scikit-learn, XGBoost

MLOps: DVC

API: FastAPI

Container: Docker

Orchestration: Kubernetes

ML Serving: Seldon Core

CI/CD: GitHub Actions

🔁 End-to-End Workflow
Raw Data → DVC → Feature Engineering → Model Training
        → Model Versioning → Docker Image
        → Kubernetes / Seldon Deployment
        → FastAPI REST API → Prediction

🧪 Feature Engineering

Implemented in feature_definitions.py:

📍 Geospatial Features

Haversine distance

Manhattan distance

Direction / bearing

⏱️ Time Features

Hour, weekday, minute

Cyclical encoding (sin/cos)

Time elapsed since epoch

🔄 Transformations

Log-scaled distance

Numeric encoding for flags

🚀 How to Run (Local)
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run data pipeline with DVC
dvc init
dvc repro

3️⃣ Train model
python src/models/train_model.py

4️⃣ Start FastAPI server
uvicorn service:app --host 0.0.0.0 --port 8080


Test:

GET  /health
POST /predict

🐳 Docker
Build & Run
docker build -t trip-duration:v1 .
docker run -p 8080:8080 trip-duration:v1

☸️ Kubernetes Deployment
kubectl apply -f deployment.yaml


Port forward:

kubectl port-forward svc/trip-duration 8080:8080

🧠 Seldon Core (Optional)

Deploy using:

kubectl apply -f seldon_deployment.yaml


This wraps the FastAPI model as a production ML service with Seldon.

🤖 CI/CD (GitHub Actions)

On every push to main:

✅ Code checkout

✅ Python dependency install

✅ Basic checks

✅ Docker image build
