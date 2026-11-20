"""Configuración del proyecto.

Cada valor se puede cambiar con una variable de entorno del mismo nombre,
sin editar el código. Ejemplo (PowerShell): $env:CONFIDENCE_THRESHOLD = "0.7"
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Modelo ---
WEIGHTS_PATH = Path(os.environ.get('WEIGHTS_PATH', BASE_DIR / 'weights.pt'))
INPUT_SIZE = int(os.environ.get('INPUT_SIZE', 640))  # El modelo se entrenó con imágenes estiradas a 640x640
CLASS_NAMES = ["Paso 1", "Paso 2", "Paso 3", "Paso 4", "Paso 5", "Paso 6"]

# --- Detección ---
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.6))  # Confianza mínima para aceptar una detección
DETECTION_HISTORY_SIZE = int(os.environ.get('DETECTION_HISTORY_SIZE', 5))  # Cuadros usados para suavizar la predicción

# --- Servidor ---
HOST = os.environ.get('HOST', '0.0.0.0')
PORT = int(os.environ.get('PORT', 5000))
# El modo debug expone el depurador de Werkzeug; solo activarlo en desarrollo.
DEBUG = os.environ.get('FLASK_DEBUG', '0') == '1'
