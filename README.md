🚕 NYC Taxi Trip Duration
End-to-End MLOps Project

This project implements a production-ready end-to-end MLOps pipeline for predicting NYC Taxi Trip Duration, including data processing, feature engineering, model training, versioning, containerization, CI/CD, and deployment using Docker, Kubernetes, and Seldon Core.


📌 Problem Statement

Given pickup and drop-off locations along with time information, predict the expected taxi trip duration (in seconds).

This is a regression problem based on the NYC Taxi Trip Duration dataset.

✨ Key Highlights

✅ Modular data pipeline (Cookiecutter style)

✅ Advanced feature engineering

✅ XGBoost model for regression

✅ DVC for data & model versioning

✅ FastAPI for model serving

✅ Docker containerization

✅ Kubernetes deployment

✅ Seldon Core (optional)

✅ GitHub Actions CI pipeline


End-to-End-MLOps-Pipeline-for-NYC-Taxi-Trip-Duration-Prediction
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
├── dvc.yaml
├── Dockerfile
├── deployment.yaml
├── seldon_deployment.yaml
├── data
│   ├── raw
│   └── processed
├── docs
├── notebooks
├── references
├── reports
│   └── figures
├── src
│   ├── data
│   │   └── make_dataset.py
│   ├── features
│   │   └── feature_definitions.py
│   ├── models
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization
└── .github
    └── workflows
        └── ci.yml
