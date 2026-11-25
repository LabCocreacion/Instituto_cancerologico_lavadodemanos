#!/usr/bin/env bash
# Elimina "Lavado de Manos" y sus accesos directos.
set -uo pipefail

APP=LavadoManos
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"

rm -f "$HOME/.local/share/applications/$APP.desktop" "$DESKTOP_DIR/$APP.desktop"
rm -rf "$HOME/.local/opt/$APP"
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true

echo "Lavado de Manos fue desinstalado."
