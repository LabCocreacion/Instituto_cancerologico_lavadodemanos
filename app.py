import base64
import os

import cv2
import numpy as np
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Configuración ---
WEIGHTS_PATH = 'weights.pt'
INPUT_SIZE = 640                # El modelo se entrenó con imágenes estiradas a 640x640
CONFIDENCE_THRESHOLD = 0.6      # Confianza mínima para aceptar una detección
DETECTION_HISTORY_SIZE = 5      # Cuadros usados para suavizar la predicción
CLASS_NAMES = ["Paso 1", "Paso 2", "Paso 3", "Paso 4", "Paso 5", "Paso 6"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO(WEIGHTS_PATH)
print("Loaded model with ultralytics YOLO")


def class_label(class_id):
    return CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"


def detect(frame):
    """Ejecuta el modelo y devuelve las cajas en coordenadas del cuadro original."""
    height, width = frame.shape[:2]
    input_frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    # verbose (por defecto) imprime en consola lo que detecta el modelo en cada cuadro
    result = model.predict(input_frame, device=device)[0]

    scale = np.array([width / INPUT_SIZE, height / INPUT_SIZE] * 2)
    boxes = []
    for x1, y1, x2, y2, conf, cls in result.boxes.data.cpu().numpy():
        x1, y1, x2, y2 = (np.array([x1, y1, x2, y2]) * scale).astype(int)
        class_id = int(cls)
        boxes.append({
            'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
            'class': class_id,
            'label': class_label(class_id),
            'confidence': float(conf),
        })
    return boxes


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


# Últimas detecciones, para suavizar el resultado entre cuadros
last_detections = []


@socketio.on('video_frame')
def handle_video_frame(data):
    global last_detections

    try:
        # Decodificar la imagen base64
        image_bytes = base64.b64decode(data['image'].split(',')[1])
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        bounding_boxes = detect(frame)

        # Dibujar las cajas sobre una copia del cuadro
        display_frame = frame.copy()
        for box in bounding_boxes:
            cv2.rectangle(display_frame, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{box['label']}: {box['confidence']:.2f}",
                        (box['x1'], box['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Detección con mayor confianza de este cuadro
        best = max(bounding_boxes, key=lambda b: b['confidence'], default=None)
        if best and best['confidence'] > CONFIDENCE_THRESHOLD:
            last_detections.append((best['class'], best['confidence']))
        else:
            last_detections.append((None, 0))
        last_detections = last_detections[-DETECTION_HISTORY_SIZE:]

        # Paso más frecuente en el historial reciente
        valid_detections = [(c, conf) for c, conf in last_detections if c is not None]
        class_id, confidence = None, 0
        if valid_detections:
            class_counts = {}
            for c, _ in valid_detections:
                class_counts[c] = class_counts.get(c, 0) + 1
            class_id = max(class_counts.items(), key=lambda x: x[1])[0]
            confidence = max(conf for c, conf in valid_detections if c == class_id)

        _, buffer = cv2.imencode('.jpg', display_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        if class_id is not None:
            step_name = class_label(class_id)
            print(f"Detected: {step_name} with confidence {confidence:.2f}")
            emit('prediction_result', {
                'step': class_id,
                'step_name': step_name,
                'confidence': float(confidence),
                'image': f'data:image/jpeg;base64,{img_base64}',
                'bounding_boxes': bounding_boxes
            })
        else:
            emit('prediction_result', {
                'step': -1,
                'step_name': 'No step detected',
                'confidence': 0,
                'image': f'data:image/jpeg;base64,{img_base64}',
                'bounding_boxes': []
            })

    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        emit('prediction_result', {'step': -1, 'step_name': 'Error', 'confidence': 0})


if __name__ == '__main__':
    # El modo debug expone el depurador de Werkzeug; solo activarlo en desarrollo (FLASK_DEBUG=1).
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    # allow_unsafe_werkzeug: Flask-SocketIO >= 5.3 se niega a usar el servidor de
    # desarrollo de Werkzeug si no se ejecuta desde una terminal interactiva.
    socketio.run(app, debug=debug, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
