"""Lógica del modelo: carga, detección y suavizado de resultados entre cuadros."""
from collections import Counter, deque

import cv2
import numpy as np

import config


def get_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(weights_path=config.WEIGHTS_PATH):
    from ultralytics import YOLO
    return YOLO(str(weights_path))


def class_label(class_id):
    if 0 <= class_id < len(config.CLASS_NAMES):
        return config.CLASS_NAMES[class_id]
    return f"Class {class_id}"


def detect(model, frame, device=None, verbose=True):
    """Ejecuta el modelo sobre un cuadro BGR.

    El cuadro se estira a INPUT_SIZE x INPUT_SIZE (igual que en el entrenamiento)
    y las cajas se devuelven en coordenadas del cuadro original.
    verbose (por defecto) imprime en consola lo que detecta el modelo en cada cuadro.
    """
    height, width = frame.shape[:2]
    input_frame = cv2.resize(frame, (config.INPUT_SIZE, config.INPUT_SIZE))
    result = model.predict(input_frame, device=device, verbose=verbose)[0]

    scale = np.array([width / config.INPUT_SIZE, height / config.INPUT_SIZE] * 2)
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


def draw_boxes(frame, boxes):
    """Devuelve una copia del cuadro con las cajas y etiquetas dibujadas."""
    output = frame.copy()
    for box in boxes:
        cv2.rectangle(output, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2)
        cv2.putText(output, f"{box['label']}: {box['confidence']:.2f}",
                    (box['x1'], box['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output


class StepSmoother:
    """Suaviza el paso detectado usando las últimas N detecciones.

    Devuelve el paso más frecuente del historial reciente (y su mayor confianza),
    para evitar saltos entre pasos por un cuadro mal clasificado.
    """

    def __init__(self, history_size=config.DETECTION_HISTORY_SIZE,
                 threshold=config.CONFIDENCE_THRESHOLD):
        self.threshold = threshold
        self.history = deque(maxlen=history_size)

    def update(self, boxes):
        """Agrega las cajas de un cuadro y devuelve (class_id, confianza) o (None, 0)."""
        best = max(boxes, key=lambda b: b['confidence'], default=None)
        if best and best['confidence'] > self.threshold:
            self.history.append((best['class'], best['confidence']))
        else:
            self.history.append((None, 0))

        valid = [(c, conf) for c, conf in self.history if c is not None]
        if not valid:
            return None, 0

        # En empate gana el paso que apareció primero en el historial
        class_id = Counter(c for c, _ in valid).most_common(1)[0][0]
        confidence = max(conf for c, conf in valid if c == class_id)
        return class_id, confidence
