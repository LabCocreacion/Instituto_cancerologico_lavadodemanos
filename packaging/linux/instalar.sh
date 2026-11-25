#!/usr/bin/env bash
# Instala "Lavado de Manos" para el usuario actual (no requiere contraseña de administrador).
# Crea un ícono en el escritorio y en el menú de aplicaciones.
set -euo pipefail

APP=LavadoManos
NAME="Lavado de Manos"
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/.local/opt/$APP"
APPS_DIR="$HOME/.local/share/applications"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"

notify() {
    if command -v zenity >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
        zenity --info --title="$NAME" --text="$1" --width=360 2>/dev/null || true
    fi
    echo -e "$1"
}

[ -x "$SRC/app/$APP" ] || { echo "No se encontró $SRC/app/$APP. Ejecute este script desde la carpeta del paquete." >&2; exit 1; }

# Copiar la aplicación (reemplaza una instalación anterior)
rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$SRC/app/." "$DEST/"
cp "$SRC/icono.png" "$DEST/icono.png"
cp "$SRC/desinstalar.sh" "$DEST/desinstalar.sh"
chmod +x "$DEST/$APP" "$DEST/desinstalar.sh"

# Acceso directo
LAUNCHER="[Desktop Entry]
Type=Application
Name=$NAME
Comment=Verificación de los pasos del lavado de manos con la cámara
Exec=$DEST/$APP
Icon=$DEST/icono.png
Terminal=true
Categories=Utility;
"
mkdir -p "$APPS_DIR"
printf '%s' "$LAUNCHER" > "$APPS_DIR/$APP.desktop"
chmod +x "$APPS_DIR/$APP.desktop"

if [ -d "$DESKTOP_DIR" ]; then
    printf '%s' "$LAUNCHER" > "$DESKTOP_DIR/$APP.desktop"
    chmod +x "$DESKTOP_DIR/$APP.desktop"
    # Marca el ícono como confiable para que se pueda abrir con doble clic
    gio set "$DESKTOP_DIR/$APP.desktop" metadata::trusted true 2>/dev/null || true
fi

update-desktop-database "$APPS_DIR" 2>/dev/null || true

notify "Instalación completa.\n\nAbra la aplicación con doble clic en el ícono \"$NAME\" del escritorio o búsquela en el menú de aplicaciones."
