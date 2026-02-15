MLOps Cats vs Dogs — End-to-End Pipeline
Project Overview

This project demonstrates a complete MLOps pipeline for a Cats vs Dogs image classification model using:

PyTorch (model training)

MLflow (experiment tracking)

DVC (data versioning)

FastAPI (model serving)

Docker (containerization)

GitHub Actions (CI/CD automation)

The goal is to simulate a production-ready ML workflow, from training to deployment and monitoring.

Architecture Flow
Git Push
   ↓
GitHub Actions (CI)
   - Install dependencies
   - Run tests (pytest)
   - Build Docker image
   ↓
CD Pipeline
   - Deploy using docker-compose
   ↓
FastAPI Inference Service
   - Health endpoint
   - Prediction endpoint
   - Monitoring logs

Project Structure
mlops-cats-dogs/
│
├── src/
│   ├── training/
│   │   ├── train.py
│   │   ├── model.py
│   │   └── preprocess.py
│   │
│   └── inference/
│       ├── app.py
│       └── predict.py
│
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md

Setup Instructions
Create Virtual Environment
python -m venv mlops2_venv
mlops2_venv\Scripts\activate   # Windows

Install Dependencies
pip install -r requirements.txt

Run Training Pipeline
python src/training/train.py


This will:

Train ResNet18 model

Log metrics in MLflow

Save model artifact

Start MLflow UI:

mlflow ui


Open:

http://127.0.0.1:5000

Run FastAPI Locally
uvicorn src.inference.app:app --reload


Open API docs:

http://127.0.0.1:8000/docs


Endpoints:

GET  /health
POST /predict

Docker Usage
Build Image
docker build -t catsdogs-api .

Run Container
docker run -p 8000:8000 catsdogs-api

Docker Compose Deployment
docker compose up


This simulates the CD deployment step.

CI/CD Pipeline

GitHub Actions automatically runs on push:


CI (Continuous Integration)

Install dependencies

Run pytest tests

Build Docker image


CD (Continuous Deployment)

Deploy container using docker-compose

Workflow file:

.github/workflows/ci.yml


Monitoring & Logging

The API includes basic monitoring features:

Request counter

Latency measurement

Structured logging

Example logs:

Received prediction request
Prediction result=Dog latency=25ms
Total requests served: 3

Run Tests
pytest


Tests include:

Preprocessing validation

Prediction logic

Error handling


Tools & Technologies

Python

PyTorch

FastAPI

MLflow

DVC

Docker

Docker Compose

GitHub Actions
