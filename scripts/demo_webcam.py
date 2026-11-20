"""Demo local del modelo con la webcam (sin interfaz web).

Uso (desde la raíz del proyecto): python scripts/demo_webcam.py
Presionar 'q' para salir.
"""
import sys
from pathlib import Path
from time import time

import cv2

# Permite importar los módulos de la raíz del proyecto al ejecutar este archivo directamente
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import detector  # noqa: E402


def process_video(video_path=0, output_path=None):
    """Muestra las detecciones sobre un video o la webcam (video_path=0).

    Si se indica output_path, guarda el video anotado en ese archivo.
    """
    device = detector.get_device()
    print(f"Using device: {device}")
    model = detector.load_model()

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
        boxes = detector.detect(model, frame, device=device)
        annotated_frame = detector.draw_boxes(frame, boxes)
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
