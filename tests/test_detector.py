"""Pruebas de detector.py que no requieren cámara ni cargar el modelo real."""
import numpy as np
import pytest

import config
import detector


class FakeTensor:
    def __init__(self, rows):
        self._rows = np.array(rows, dtype=np.float32).reshape(-1, 6)

    def cpu(self):
        return self

    def numpy(self):
        return self._rows


class FakeModel:
    """Imita model.predict de ultralytics devolviendo cajas fijas (x1, y1, x2, y2, conf, cls)."""

    def __init__(self, rows):
        self.rows = rows
        self.received_shape = None

    def predict(self, frame, device=None, verbose=True):
        self.received_shape = frame.shape

        class Boxes:
            data = FakeTensor(self.rows)

        class Result:
            boxes = Boxes()

        return [Result()]


def box(class_id, confidence):
    return {'class': class_id, 'confidence': confidence}


# --- detect ---

def test_detect_estira_el_cuadro_al_tamano_de_entrada():
    model = FakeModel([])
    detector.detect(model, np.zeros((480, 640, 3), np.uint8))
    assert model.received_shape[:2] == (config.INPUT_SIZE, config.INPUT_SIZE)


def test_detect_devuelve_cajas_en_coordenadas_del_cuadro_original():
    # Caja en la esquina inferior derecha del cuadro de 640x640 que recibe el modelo
    model = FakeModel([[320, 320, 640, 640, 0.9, 2]])
    boxes = detector.detect(model, np.zeros((480, 1280, 3), np.uint8))
    assert boxes == [{
        'x1': 640, 'y1': 240, 'x2': 1280, 'y2': 480,
        'class': 2, 'label': 'Paso 3', 'confidence': pytest.approx(0.9),
    }]


def test_detect_sin_cajas():
    assert detector.detect(FakeModel([]), np.zeros((480, 640, 3), np.uint8)) == []


def test_class_label_fuera_de_rango():
    assert detector.class_label(0) == 'Paso 1'
    assert detector.class_label(99) == 'Class 99'


# --- StepSmoother ---

def test_smoother_sin_detecciones():
    assert detector.StepSmoother().update([]) == (None, 0)


def test_smoother_ignora_confianza_bajo_el_umbral():
    smoother = detector.StepSmoother(threshold=0.6)
    assert smoother.update([box(1, 0.5)]) == (None, 0)


def test_smoother_usa_la_caja_de_mayor_confianza_del_cuadro():
    smoother = detector.StepSmoother(threshold=0.6)
    assert smoother.update([box(1, 0.7), box(3, 0.95)]) == (3, 0.95)


def test_smoother_devuelve_el_paso_mas_frecuente():
    smoother = detector.StepSmoother(history_size=5, threshold=0.6)
    smoother.update([box(2, 0.9)])
    smoother.update([box(2, 0.8)])
    # Un cuadro mal clasificado no cambia el paso detectado
    assert smoother.update([box(4, 0.99)]) == (2, 0.9)


def test_smoother_olvida_detecciones_fuera_del_historial():
    smoother = detector.StepSmoother(history_size=2, threshold=0.6)
    smoother.update([box(1, 0.9)])
    smoother.update([])
    smoother.update([])
    assert smoother.update([]) == (None, 0)


def test_smoothers_independientes_no_se_mezclan():
    a = detector.StepSmoother(threshold=0.6)
    b = detector.StepSmoother(threshold=0.6)
    a.update([box(1, 0.9)])
    assert b.update([]) == (None, 0)
