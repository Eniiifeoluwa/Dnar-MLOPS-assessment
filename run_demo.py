"""
One-command demo script for ML Inference Service.

Steps:
1. Build Docker image
2. Run container locally
3. Wait for service to be ready
4. Execute test_experiment.py
5. Cleanup
"""
import subprocess
import time
import requests
import os
import signal

IMAGE_NAME = "ml-inference-service:latest"
CONTAINER_NAME = "ml-inference-demo"
API_URL = "http://localhost:8000"


def build_image():
    print("🔹 Building Docker image...")
    subprocess.run(["docker", "build", "-t", IMAGE_NAME, "."], check=True)
    print("Docker image built.")


def run_container():
    print("🔹 Starting Docker container...")
    # Run container in detached mode
    subprocess.run([
        "docker", "run", "--name", CONTAINER_NAME,
        "-p", "8000:8000", "-d", IMAGE_NAME
    ], check=True)
    print("Container started.")


def wait_until_ready(timeout=30):
    print("Waiting for service to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{API_URL}/ready")
            if r.status_code == 200:
                print("Service is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError("Service did not become ready in time.")


def run_tests():
    print("Running test_experiment.py...")
    subprocess.run(["python", os.path.join("scripts", "test_experiment.py")], check=True)
    print("Tests completed.")


def cleanup():
    print("Stopping and removing container...")
    subprocess.run(["docker", "stop", CONTAINER_NAME], check=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME], check=True)
    print("Cleanup done.")


if __name__ == "__main__":
    try:
        build_image()
        run_container()
        wait_until_ready()
        run_tests()
    finally:
        cleanup()
