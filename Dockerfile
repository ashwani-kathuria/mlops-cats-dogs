# ---- Base Image ----
FROM python:3.11-slim

ENV DEPLOY_ENV=local

# ---- Set working directory ----
WORKDIR /app

# ---- Copy requirements ----
COPY requirements.txt .

# ---- Install dependencies ----
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ---- Copy project files ----
COPY . .
 
# ---- Expose FastAPI port ----
EXPOSE 8000
 
# ---- Start API server ----
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
