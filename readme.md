# 🐱🐶 MLOps Cats vs Dogs — End-to-End Production Pipeline

An end-to-end MLOps project demonstrating **model training, experiment tracking, API deployment, CI/CD automation, monitoring, and AWS cloud deployment** using modern industry tools.

---

# 🚀 Project Overview

This repository implements a complete Machine Learning lifecycle:

✅ Model Training using PyTorch (ResNet18 Transfer Learning)
✅ Experiment Tracking with MLflow
✅ Data Versioning using DVC
✅ REST API using FastAPI
✅ Containerization with Docker
✅ CI/CD using GitHub Actions
✅ Deployment to AWS App Runner
✅ Monitoring using Prometheus + Grafana

The goal is to simulate a **production-ready MLOps architecture**.

---

# 🧱 Tech Stack

| Area                | Tools Used           |
| ------------------- | -------------------- |
| Model Training      | PyTorch, ResNet18    |
| Experiment Tracking | MLflow               |
| Data Versioning     | DVC                  |
| API                 | FastAPI              |
| Documentation UI    | Swagger              |
| Containerization    | Docker               |
| CI/CD               | GitHub Actions       |
| Cloud Deployment    | AWS App Runner + ECR |
| Monitoring          | Prometheus + Grafana |

---

# 📁 Project Structure

```
src/
 ├── training/
 │    ├── train.py
 │    ├── model.py
 │    └── preprocess.py
 ├── inference/
 │    ├── app.py
 │    └── predict.py
tests/
Dockerfile
docker-compose.yml
requirements.txt
```

---

# ⚙️ Local Setup

## 1️⃣ Clone Repository

```
git clone <repo-url>
cd mlops-cats-dogs
```

## 2️⃣ Create Virtual Environment

```
python -m venv mlops2_venv
mlops2_venv\Scripts\activate
pip install -r requirements.txt
```

---

# 🧠 Model Training

Start MLflow UI:

```
mlflow ui
```

Run training:

```
python src/training/train.py
```

Open:

```
http://127.0.0.1:5000
```

You will see:

* Parameters
* Metrics
* Artifacts
* Model versions

---

# 🌐 Run API Locally

```
uvicorn src.inference.app:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

Upload an image and test prediction.

---

# 🐳 Docker Usage

## Build Image

```
docker build -t catsdogs-api .
```

## Run Container

```
docker run -p 8000:8000 catsdogs-api
```

---

# 📊 Monitoring (Prometheus + Grafana)

Start monitoring stack:

```
docker compose up
```

Access dashboards:

```
API:        http://localhost:8000/docs
Prometheus: http://localhost:9090
Grafana:    http://localhost:3000
```


# 🔄 CI/CD Pipeline

GitHub Actions automatically:

* Runs unit tests
* Builds Docker image
* Pushes image to AWS ECR
* Triggers deployment on AWS App Runner


# ☁️ AWS Deployment

Deployment uses:

* Amazon ECR — container registry
* AWS App Runner — serverless container hosting

Environment variable:

```
DEPLOY_ENV=aws
```

Public endpoint is generated automatically after deployment.

---

# 🔐 Security

* GitHub OIDC used instead of static AWS keys
* IAM Role authentication for ECR push
* Secrets managed via GitHub Secrets

---

# 🧪 Testing

Run tests locally:

```
pytest
```

Smoke tests validate preprocessing and prediction pipeline.

---

# 🧩 Future Improvements

* ECS Fargate deployment
* Model Registry integration
* Canary deployments
* Auto-scaling dashboards
* Model performance monitoring

---

# 👨‍💻 Author

**Ashwani Kathuria**
MLOps | AI Engineering | Backend Systems

---

⭐ If you find this project useful, feel free to star the repository!
