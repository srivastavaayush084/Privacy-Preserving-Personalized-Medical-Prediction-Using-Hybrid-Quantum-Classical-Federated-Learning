import sys
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting non-torch check...")

try:
    log("Importing sklearn...")
    import sklearn
    log(f"Sklearn version: {sklearn.__version__}")
except Exception as e:
    log(f"Sklearn error: {e}")

try:
    log("Importing xgboost...")
    import xgboost
    log(f"XGBoost version: {xgboost.__version__}")
except Exception as e:
    log(f"XGBoost error: {e}")

try:
    log("Importing pennylane...")
    import pennylane as qml
    log(f"PennyLane version: {qml.__version__}")
except Exception as e:
    log(f"PennyLane error: {e}")

try:
    log("Importing flwr...")
    import flwr
    log(f"Flower version: {flwr.__version__}")
except Exception as e:
    log(f"Flower error: {e}")

try:
    log("Importing streamlit...")
    import streamlit
    log(f"Streamlit version: {streamlit.__version__}")
except Exception as e:
    log(f"Streamlit error: {e}")

log("Non-torch check completed!")
