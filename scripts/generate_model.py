import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load Iris dataset
data = load_iris()
X, y = data.data, data.target

# Train simple classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")
