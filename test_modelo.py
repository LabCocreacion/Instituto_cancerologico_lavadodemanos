"""Prueba local del modelo con la webcam (sin interfaz web).

Uso: python test_modelo.py   — presionar 'q' para salir.
"""
from time import time

import cv2
import torch
from ultralytics import YOLO

WEIGHTS_PATH = 'weights.pt'
INPUT_SIZE = 640  # El modelo se entrenó con imágenes estiradas a 640x640

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO(WEIGHTS_PATH)


def process_video(video_path=0, output_path=None):
    """Muestra las detecciones sobre un video o la webcam (video_path=0).

    Si se indica output_path, guarda el video anotado en ese archivo.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time()
        input_frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        # verbose (por defecto) imprime en consola los pasos reconocidos en cada cuadro
        result = model.predict(input_frame, device=device)[0]
        annotated_frame = cv2.resize(result.plot(), (width, height))
        inference_time = time() - start_time

        fps_text = f"Inference time: {inference_time:.2f}s ({1 / inference_time:.2f} FPS)"
        cv2.putText(annotated_frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Hand Washing Detection', annotated_frame)
        if writer:
            writer.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video(0)
