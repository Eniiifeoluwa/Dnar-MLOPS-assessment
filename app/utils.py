import logging
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total inference requests', ['status', 'endpoint'])
REQUEST_LATENCY = Histogram('inference_request_duration_seconds', 'Request latency', buckets=[0.01,0.05,0.1,0.5,1.0,2.0,5.0])
PREDICTION_DISTRIBUTION = Histogram('prediction_value_distribution', 'Prediction distribution', buckets=[0,0.25,0.5,0.75,1.0])
MODEL_VERSION = Gauge('model_version_info', 'Model version info')

# Middleware helper
async def log_request_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} {response.status_code} {duration:.3f}s")
    return response

# Metrics endpoint helper
def prometheus_metrics():
    return Response(generate_latest(), media_type="text/plain")
