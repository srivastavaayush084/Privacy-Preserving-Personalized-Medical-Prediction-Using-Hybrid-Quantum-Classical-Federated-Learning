import sys
import time

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Starting environment check...")

try:
    log("Importing torch...")
    import torch
    log(f"Torch imported: {torch.__version__}")
except Exception as e:
    log(f"Torch error: {e}")

try:
    log("Importing pennylane...")
    import pennylane as qml
    log(f"PennyLane imported: {qml.__version__}")
except Exception as e:
    log(f"PennyLane error: {e}")

try:
    log("Importing flwr...")
    import flwr
    log(f"Flower imported: {flwr.__version__}")
except Exception as e:
    log(f"Flower error: {e}")

try:
    log("Importing opacus...")
    import opacus
    log(f"Opacus imported: {opacus.__version__}")
except Exception as e:
    log(f"Opacus error: {e}")

try:
    log("Importing xgboost...")
    import xgboost
    log(f"XGBoost imported: {xgboost.__version__}")
except Exception as e:
    log(f"XGBoost error: {e}")

try:
    log("Importing shap...")
    import shap
    log(f"SHAP imported: {shap.__version__}")
except Exception as e:
    log(f"SHAP error: {e}")

try:
    log("Importing streamlit...")
    import streamlit
    log(f"Streamlit imported: {streamlit.__version__}")
except Exception as e:
    log(f"Streamlit error: {e}")

log("All checks completed successfully!")
