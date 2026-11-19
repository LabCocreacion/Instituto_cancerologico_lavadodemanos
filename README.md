# Detección de Lavado de Manos con YOLOv11

Aplicación web que verifica en tiempo real, desde la cámara del navegador, que una persona realice los **6 pasos del protocolo de lavado de manos**. Fue desarrollada para el **Instituto Nacional de Cancerología** como apoyo al control de acceso a la UCI y al cumplimiento de los protocolos sanitarios para el cuidado de los pacientes.

https://github.com/user-attachments/assets/9d6c0db9-fd37-438b-b22e-68a6f4005ff8

- 📄 Informe completo del proceso: [INC1.pdf](./INC1.pdf)
- 🧠 Detalles de la arquitectura YOLOv11: [artículo en Medium](https://medium.com/@nikhil-rao-20/yolov11-explained-next-level-object-detection-with-enhanced-speed-and-accuracy-2dbe2d376f71)

---

## Contenido

- [Cómo funciona](#cómo-funciona)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Configuración](#configuración)
- [Modelo](#modelo)
- [Licencia](#licencia)
- [Referencias](#referencias)

---

## Cómo funciona

```
Navegador (React)  ──frame base64──▶  Flask + Socket.IO  ──▶  YOLOv11 (weights.pt)
       ▲                                                              │
       └──────────── paso detectado + confianza + cajas ◀─────────────┘
```

1. La interfaz web (`templates/index.html` + `static/js/app.js`) captura la cámara y envía cuadros al servidor por **WebSocket** (evento `video_frame`).
2. El servidor (`app.py`) ejecuta el modelo sobre cada cuadro.
3. Para evitar saltos entre predicciones, se suaviza el resultado con el historial de las últimas 5 detecciones y se descartan las que tengan confianza menor a 0.6.
4. El servidor responde (evento `prediction_result`) con el paso detectado, la confianza, la imagen anotada y las cajas delimitadoras; la interfaz va marcando el progreso de los 6 pasos.

## Estructura del proyecto

```
.
├── app.py               # Servidor Flask + Socket.IO e inferencia en tiempo real
├── test_modelo.py       # Script local para probar el modelo con la webcam (OpenCV)
├── weights.pt           # Pesos del modelo YOLOv11 entrenado
├── requirements.txt     # Dependencias de Python
├── templates/
│   └── index.html       # Página principal
├── static/
│   ├── js/app.js        # Interfaz en React (captura de cámara y progreso)
│   └── images/          # Ilustraciones de los pasos 1 a 6
├── imagenes/            # Gráficas de entrenamiento usadas en este README
└── INC1.pdf             # Informe técnico del proyecto
```

## Requisitos

- Python 3.10 o superior
- Cámara web
- Navegador moderno (Chrome, Edge o Firefox) con permiso de acceso a la cámara
- *(Opcional)* GPU NVIDIA con CUDA: se usa automáticamente si está disponible; si no, la inferencia corre en CPU

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/LabCocreacion/Instituto_cancerologico_lavadodemanos.git
cd Instituto_cancerologico_lavadodemanos

# 2. Crear y activar un entorno virtual
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

> El entorno virtual no se versiona (está en `.gitignore`); cada persona debe crear el suyo.

## Uso

### Aplicación web

```bash
python app.py
```

Abrir [http://localhost:5000](http://localhost:5000) en el navegador y permitir el acceso a la cámara. El servidor escucha en `0.0.0.0:5000`, así que también es accesible desde otros equipos de la misma red (`http://<ip-del-equipo>:5000`).

> Los navegadores solo permiten usar la cámara en `localhost` o bajo HTTPS. Para acceder desde otro equipo de la red se necesita servir la aplicación con HTTPS.

### Prueba local con la webcam (sin interfaz web)

```bash
python test_modelo.py
```

Abre una ventana de OpenCV con las detecciones sobre la imagen de la cámara. Presionar `q` para salir.

## Configuración

Los parámetros principales están al inicio de `app.py`:

| Parámetro               | Valor por defecto | Descripción                                               |
|-------------------------|-------------------|-----------------------------------------------------------|
| `weights_path`          | `weights.pt`      | Ruta a los pesos del modelo                               |
| `CONFIDENCE_THRESHOLD`  | `0.6`             | Confianza mínima para aceptar una detección               |
| `DETECTION_HISTORY_SIZE`| `5`               | Número de cuadros usados para suavizar la predicción      |
| `CLASS_NAMES`           | `Paso 1` … `Paso 6` | Nombres de las clases que se muestran en la interfaz    |

## Modelo

El modelo se entrenó con la arquitectura **YOLOv11** en **Roboflow 3.0** (PyTorch). Los pesos están en `weights.pt` en la raíz del repositorio.

### Métricas

| Métrica   | Valor  |
|-----------|--------|
| mAP       | 99.3 % |
| Precisión | 99.0 % |
| Recall    | 99.5 % |
| mAP50     | ~99.5 % |
| mAP50-95  | ~78 %  |

![Métricas del modelo](imagenes/graficas_entrenamiento.png)

### Conjunto de datos

- **Imágenes originales:** 2,975
- **Imágenes tras aumentación:** 7,788
- **División:** 80 % entrenamiento · 10 % validación · 10 % prueba

**Preprocesamiento**

- Auto-orientación
- Redimensionamiento (estirado) a 640×640 px
- Conversión a escala de grises, para reducir la variación de color y enfocarse en la forma

**Aumentación** (3 salidas por ejemplo de entrenamiento)

- Rotación aleatoria entre −8° y +8°
- Exposición aleatoria entre −8 % y +8 %

### Entrenamiento

![Gráficas de la función de error](imagenes/graficas_funcion_error.png)

| Pérdida    | Inicio | Final      |
|------------|--------|------------|
| Box Loss   | ~2.2   | ~0.8       |
| Class Loss | ~3.0   | ~0.2       |
| DFL Loss   | ~1.8   | ~1.2 – 1.3 |

El mAP mejora rápidamente en las primeras épocas, supera el 90 % hacia la época 20 y sigue subiendo de forma gradual hasta llegar a 99.3 % al final del entrenamiento.

## Licencia

Este software fue desarrollado por Tomás Barón, Samuel Ramírez, Giovanni Moreno y Santiago Toledo-Cortés, de la Facultad de Ingeniería de la Universidad de La Sabana, y se otorga al Instituto Nacional de Cancerología un derecho de uso exclusivo. El software no puede ser redistribuido, modificado ni utilizado por terceros sin el consentimiento expreso de los desarrolladores. Cualquier otra entidad interesada en usar esta tecnología debe contactar a Santiago Toledo-Cortés (santiagotoco@unisabana.edu.co).

## Referencias

- [Roboflow](https://roboflow.com/)
- [Ultralytics](https://www.ultralytics.com/)
- [PyTorch](https://pytorch.org/)
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/)
