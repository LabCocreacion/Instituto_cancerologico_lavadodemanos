"""Punto de entrada del ejecutable para Linux.

Arranca el servidor y abre la aplicación en el navegador predeterminado.
Si la aplicación ya está corriendo, solo abre el navegador.
"""
import os
import socket
import subprocess
import sys
import threading
import time

# En el dispositivo la app se usa localmente: por defecto solo escucha en este equipo.
os.environ.setdefault('HOST', '127.0.0.1')
# Evita que ultralytics intente conectarse a internet
os.environ.setdefault('YOLO_OFFLINE', '1')

import config  # noqa: E402

URL = f"http://localhost:{config.PORT}"


def server_running():
    try:
        with socket.create_connection(('127.0.0.1', config.PORT), timeout=1):
            return True
    except OSError:
        return False


def open_browser():
    # El ejecutable de PyInstaller modifica LD_LIBRARY_PATH; se restaura el original
    # para que el navegador del sistema no cargue las librerías del paquete.
    env = dict(os.environ)
    original = env.pop('LD_LIBRARY_PATH_ORIG', None)
    if original is not None:
        env['LD_LIBRARY_PATH'] = original
    else:
        env.pop('LD_LIBRARY_PATH', None)
    try:
        subprocess.Popen(['xdg-open', URL], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        print(f"Abra manualmente {URL} en el navegador.")


def open_browser_when_ready():
    for _ in range(240):  # hasta 2 minutos mientras carga el modelo
        if server_running():
            open_browser()
            return
        time.sleep(0.5)
    print(f"El servidor no respondió a tiempo. Intente abrir {URL} manualmente.")


def main():
    if server_running():
        print(f"La aplicación ya está en ejecución en {URL}")
        open_browser()
        return

    print("Iniciando la aplicación de lavado de manos...")
    print("Cierre esta ventana para detener la aplicación.\n")
    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    import app  # Carga el modelo (tarda unos segundos)
    app.socketio.run(app.app, host=config.HOST, port=config.PORT,
                     allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    sys.exit(main())
