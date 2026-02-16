import joblib
import logging

logger = logging.getLogger(__name__)

class ModelHolder:
    """Singleton for ML model"""
    def __init__(self, path="model.joblib", version="1.0.0"):
        self.model = None
        self.version = version
        self.path = path

    def load(self):
        try:
            self.model = joblib.load(self.path)
            logger.info(f"Model loaded. Version: {self.version}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, features):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.predict(features)

    def predict_proba(self, features):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        try:
            return self.model.predict_proba(features)
        except AttributeError:
            return None


model_holder = ModelHolder()
