"""Servidor web: recibe cuadros de la cámara por Socket.IO y responde con el paso detectado."""
import base64
import traceback

import cv2
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

import config
import detector

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*")

device = detector.get_device()
print(f"Using device: {device}")
model = detector.load_model()
print("Loaded model with ultralytics YOLO")

# Historial de detecciones de cada cliente conectado (clave: id de la conexión)
smoothers = {}


def decode_frame(data_url):
    """Convierte una imagen 'data:image/jpeg;base64,...' en un cuadro BGR de OpenCV."""
    image_bytes = base64.b64decode(data_url.split(',')[1])
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)


def encode_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    smoothers[request.sid] = detector.StepSmoother()
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    smoothers.pop(request.sid, None)
    print('Client disconnected')


@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        frame = decode_frame(data['image'])
        bounding_boxes = detector.detect(model, frame, device=device)
        image = encode_frame(detector.draw_boxes(frame, bounding_boxes))

        smoother = smoothers.setdefault(request.sid, detector.StepSmoother())
        class_id, confidence = smoother.update(bounding_boxes)

        if class_id is not None:
            step_name = detector.class_label(class_id)
            print(f"Detected: {step_name} with confidence {confidence:.2f}")
            emit('prediction_result', {
                'step': class_id,
                'step_name': step_name,
                'confidence': float(confidence),
                'image': image,
                'bounding_boxes': bounding_boxes
            })
        else:
            emit('prediction_result', {
                'step': -1,
                'step_name': 'No step detected',
                'confidence': 0,
                'image': image,
                'bounding_boxes': []
            })

    except Exception as e:
        print(f"Error processing frame: {e}")
        traceback.print_exc()
        emit('prediction_result', {'step': -1, 'step_name': 'Error', 'confidence': 0})


if __name__ == '__main__':
    # allow_unsafe_werkzeug: Flask-SocketIO >= 5.3 se niega a usar el servidor de
    # desarrollo de Werkzeug si no se ejecuta desde una terminal interactiva.
    socketio.run(app, debug=config.DEBUG, host=config.HOST, port=config.PORT,
                 allow_unsafe_werkzeug=True)
