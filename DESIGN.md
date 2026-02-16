## **DESIGN.md**

```markdown
# ML Inference Service — Design & Decisions

## 1. Architecture Overview

A minimal production-style ML inference service using FastAPI and a pre-trained Iris classifier.

**Components:**

- **FastAPI** — REST API for predictions, health, readiness, and metrics
- **ML Model** — Pre-trained RandomForestClassifier on Iris dataset
- **Docker** — Containerizes the service
- **Kubernetes** — Deployment, scaling, liveness/readiness, resource limits
- **Prometheus Metrics** — Observability: request counts, latency, prediction distribution

**Flow:**

```
Client --> FastAPI API --> ML Model
|         |             |
|         |-- /health  |
|         |-- /ready   |
|         |-- /metrics |

```

1. Client sends `/predict` request with feature vector.  
2. FastAPI validates input via Pydantic schemas.  
3. Model predicts class and calculates confidence.  
4. Response includes predicted class, confidence, model version, and optional request ID.  
5. `/health` and `/ready` endpoints support liveness/readiness probes.  
6. Prometheus metrics exposed at `/metrics`.

---

## 2. Key Engineering Decisions

- **Stateless design:** Enables horizontal scaling, easy Kubernetes deployment.  
- **Simple model:** Iris dataset with RandomForestClassifier; avoids heavy training.  
- **Structured logging & metrics:** Ensures observability and production readiness.  
- **Error handling & validation:** Invalid requests return HTTP 400.  
- **Resource limits:** CPU/memory requests and limits prevent cluster overload.  
- **Exclusions:** No database, no authentication, no GPU required for Iris-scale models (it is CPU-sufficient).

---

## 3. Major Trade-offs & Exclusions

**Trade-offs made:**

- Trained a simple Iris model to focus on service design and deployment.
- **Stateless service**: no database or session state, prioritizing scalability over persistent storage.
- Minimal observability (Prometheus counters + histograms) rather than full APM/logging stack to keep it lightweight.
- Fixed Docker + Kubernetes configuration; no multi-model routing to keep the solution simple and minimal.

**What was intentionally not built:**

- No authentication or authorization layer: outside scope of minimal inference service.
- No GPU support for model training: Iris dataset is small and CPU is sufficient.
- No CI/CD pipeline or Terraform infrastructure.
- No database or message queue: service is stateless, predictions are synchronous.

---

## 4. Production Considerations

**Service Health:**  
- Liveness: `/health`  
- Readiness: `/ready`  
- Metrics: Prometheus counters and histograms  
- Structured logs with request IDs

**Model Performance:**  
- Track prediction distribution over time  
- Alerts for unusual distributions (drift detection)

**Rollback & Model Management:**  
- Versioned models via `model_version`  
- Replace `model.joblib` and perform Kubernetes rolling update for zero downtime

**Secrets:**  
- Environment variables or Kubernetes Secrets for API keys or DB credentials

---

## 5. Scaling & GPU Support

**Horizontal Scaling:**  
- Stateless design allows multiple replicas behind LoadBalancer  
- Kubernetes HPA can scale pods based on CPU or custom metrics  

**Multiple Models / Teams:**  
- Each model can be a separate deployment or routed via key  
- Teams can deploy isolated services for different models

**GPU Support (if needed):**  
1. Use GPU-enabled node pool in Kubernetes  
2. Use CUDA-enabled Docker image (`nvidia/cuda:12.2-runtime`)  
3. Add GPU resource request in pod spec: `nvidia.com/gpu: 1`  
4. Install GPU ML libraries (PyTorch/TensorFlow GPU)  

**Diagram:**

```
```
    +-------------+
    |   Client    |
    +------+------+
           |
    +------+------+
    | FastAPI API |
    +------+------+
           |
 +---------+---------+
 |       ML Model    |
 +---------+---------+
           |
    +------+------+
    | Prometheus   |
    +-------------+
```

```

---

## 6. Assumptions

- Minimal, production-style system  
- Single Iris model; GPU not required  
- Focus on API design, observability, deployment readiness  
- Stateless design ensures easy horizontal scaling
```
