🚕 NYC Taxi Trip Duration
End-to-End MLOps Project

This project implements a production-ready end-to-end MLOps pipeline for predicting NYC Taxi Trip Duration, covering data processing, feature engineering, model training, data & model versioning, containerization, CI/CD, and deployment using Docker, Kubernetes, and Seldon Core.

📌 Problem Statement

Given pickup and drop-off locations along with time information, predict the expected taxi trip duration (in seconds).

This is a regression problem based on the well-known NYC Taxi Trip Duration dataset.

✨ Key Highlights

✅ Modular data pipeline (Cookiecutter style)

✅ Advanced feature engineering

Distance (Haversine, Manhattan)

Direction / Bearing

Time-based features

✅ XGBoost model for regression

✅ DVC for data & model versioning

✅ FastAPI for model serving

✅ Docker containerization

✅ Kubernetes deployment

✅ Seldon Core for ML serving (optional)

✅ GitHub Actions CI pipeline

trip-duration-mlops/
├── data/
│   ├── raw/                # Raw NYC taxi CSV data
│   └── processed/          # Feature-engineered data (DVC tracked)
│
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── feature_definitions.py
│   └── models/
│       └── train_model.py
│
├── service.py              # FastAPI inference service
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Training parameters (optional)
├── Dockerfile
├── deployment.yaml         # Kubernetes deployment
├── seldon_deployment.yaml  # Seldon Core deployment
├── requirements.txt
├── .gitignore
└── .github/workflows/ci.yml

Tech Stack

Language: Python

ML: Scikit-learn, XGBoost

MLOps: DVC

API: FastAPI

Container: Docker

Orchestration: Kubernetes

ML Serving: Seldon Core

CI/CD: GitHub Actions

Raw Data
   ↓
DVC (Versioning)
   ↓
Feature Engineering
   ↓
Model Training (XGBoost)
   ↓
Model Versioning
   ↓
Docker Image
   ↓
Kubernetes / Seldon Deployment
   ↓
FastAPI REST API
   ↓
Trip Duration Prediction
