# ML Inference Service — DNAR

A **production-ready, stateless machine learning inference API** for DNAR.  
Built with **FastAPI**, **Docker**, and **Kubernetes**, it serves predictions from a pre-trained **Iris model** and exposes **health, readiness, and Prometheus metrics**.

---

## Features

- REST endpoints for `/predict`, `/health`, `/ready`, `/metrics`
- Pre-trained Iris ML model (`RandomForestClassifier`)
- Prometheus metrics for request counts, latency, and prediction distribution
- Dockerized for easy deployment
- Kubernetes manifests with **resource limits**, **liveness**, and **readiness probes**
- Stateless design allows horizontal scaling

---

## Repository Structure

```
ml-inference-service/
├── app/
│   ├── main.py            # FastAPI app
│   ├── ml_model.py        # Model loader
│   ├── models.py          # Pydantic schemas
│   └── utils.py           # Helper functions
├── scripts/
│   ├── generate_model.py  # Generates pre-trained model
│   └── test_experiment.py # Python example usage of API
├── k8/
│   └── deployment.yaml    # Kubernetes deployment + service
├── requirements.txt
├── Dockerfile
├── README.md
├── DESIGN.md
|── run_demo.py        # Optional interactive demo
└── model.joblib
```
---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Eniiifeoluwa/Dnar-MLOPS-assessment.git
cd "your base folder" for instance, here , it is ml-inference-service
```

### 2. Install dependencies (local Python environment)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3. Generate the model (if not included)

```bash
python generate_model.py
```

---

## Run Locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

- `GET /health` — Liveness probe
- `GET /ready` — Readiness probe
- `POST /predict` — ML predictions
- `GET /metrics` — Prometheus metrics

---

## Run with Docker

Build the Docker image:

```bash
docker build -t ml-inference-service:latest .
```

Run the container:

```bash
docker run -p 8000:8000 ml-inference-service:latest
```

Access the service at: `http://localhost:8000`

---

## Run with Kubernetes

Apply the deployment and service:

```bash
kubectl apply -f k8/deployment.yaml
```

- Service exposes port `80` and routes to pods on `8000`
- Liveness and readiness probes included
- Resource requests/limits: `CPU` 100m-500m, `Memory` 128Mi-512Mi

> **Note:** GPU support is optional; see DESIGN.md for configuration.

---

## Test the API

Use the provided Python script:

```bash
python scripts/test_experiment.py
```

It demonstrates:

- Health check
- Readiness check
- Predictions for Iris Setosa, Versicolor, and Virginica
- Error handling for invalid requests

---

## Observability

Prometheus metrics available at `/metrics`

Metrics include:

- `inference_requests_total` (success/error by endpoint)
- `inference_request_duration_seconds` (latency histogram)
- `prediction_value_distribution` (confidence histogram)

Optional: integrate with **Grafana** for dashboards.

---

## Assumptions

- Minimal, stateless production-style system
- Single Iris model; CPU sufficient
- Focus on API design, observability, deployment readiness
- Horizontal scaling via Kubernetes HPA

---

## License

This project is for DNAR internal use / technical assessment only.