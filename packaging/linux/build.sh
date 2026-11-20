#!/usr/bin/env bash
# Genera el ejecutable de "Lavado de Manos" para Ubuntu (x86_64 o ARM64/Jetson).
#
# Se ejecuta EN el equipo Linux donde se va a usar la aplicación (o en uno con el
# mismo Ubuntu y la misma arquitectura), desde la raíz del proyecto:
#
#     bash packaging/linux/build.sh
#
# Usa un entorno virtual propio (.venv-build). Si ese entorno no tiene PyTorch,
# instala la versión para CPU. Para un ejecutable que use la GPU de una Jetson,
# instale antes en .venv-build el PyTorch de NVIDIA (ver README, sección
# "Ejecutable para Ubuntu").
#
# Resultado: dist/LavadoManos-<arquitectura>-<cpu|gpu>.tar.gz
set -euo pipefail

APP=LavadoManos
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HERE="$ROOT/packaging/linux"
VENV="$ROOT/.venv-build"
WORK="$ROOT/build/pyinstaller"
OUT="$ROOT/dist"

say() { echo -e "\n==> $*"; }
fail() { echo -e "\nERROR: $*" >&2; exit 1; }

# --- 1. Requisitos del sistema ---
[ "$(uname -s)" = "Linux" ] || fail "Este script debe ejecutarse en Linux (Ubuntu)."
command -v python3 >/dev/null || fail "Falta Python 3. Instálelo con: sudo apt install python3"
python3 -c "import venv, ensurepip" 2>/dev/null \
    || fail "Falta el módulo venv. Instálelo con: sudo apt install python3-venv python3-dev"
command -v objdump >/dev/null \
    || fail "Falta binutils (lo necesita PyInstaller). Instálelo con: sudo apt install binutils"

# --- 2. Entorno virtual de compilación ---
if [ ! -x "$VENV/bin/python" ]; then
    say "Creando entorno virtual en .venv-build"
    python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip >/dev/null

# --- 3. PyTorch ---
if "$PY" -c "import torch" 2>/dev/null; then
    say "Se usará el PyTorch ya instalado en .venv-build ($("$PY" -c 'import torch; print(torch.__version__)'))"
    "$PY" -c "import torchvision" 2>/dev/null \
        || fail "Hay PyTorch pero falta torchvision en .venv-build. Instale el torchvision que corresponde a su PyTorch (en Jetson, el de NVIDIA) y vuelva a ejecutar."
else
    say "Instalando PyTorch para CPU (no se encontró uno en .venv-build)"
    "$PY" -m pip install --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# --- 4. Resto de dependencias ---
say "Instalando dependencias del proyecto y PyInstaller"
"$PY" -m pip install --retries 10 -r "$ROOT/requirements.txt" pyinstaller

if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    VARIANT=gpu
    say "PyTorch detecta la GPU: el ejecutable usará CUDA"
else
    VARIANT=cpu
    say "PyTorch no detecta GPU: el ejecutable usará la CPU"
fi

# --- 5. Compilar con PyInstaller ---
say "Compilando el ejecutable (puede tardar varios minutos)"
rm -rf "$WORK"
mkdir -p "$WORK" "$OUT"
"$PY" -m PyInstaller --noconfirm --clean \
    --name "$APP" \
    --workpath "$WORK/work" \
    --specpath "$WORK" \
    --distpath "$WORK/dist" \
    --paths "$ROOT" \
    --add-data "$ROOT/templates:templates" \
    --add-data "$ROOT/static:static" \
    --add-data "$ROOT/weights.pt:." \
    --collect-data ultralytics \
    --collect-submodules ultralytics \
    --hidden-import engineio.async_drivers.threading \
    --exclude-module tkinter \
    "$HERE/launcher.py"

# --- 6. Armar el paquete para el dispositivo ---
say "Armando el paquete"
PKG="$WORK/pkg/$APP"
mkdir -p "$PKG"
cp -r "$WORK/dist/$APP" "$PKG/app"
cp "$HERE/instalar.sh" "$HERE/desinstalar.sh" "$HERE/LEEME.txt" "$PKG/"
cp "$ROOT/static/images/step1.png" "$PKG/icono.png"
chmod +x "$PKG/instalar.sh" "$PKG/desinstalar.sh" "$PKG/app/$APP"

TARBALL="$OUT/${APP}-$(uname -m)-${VARIANT}.tar.gz"
tar -C "$WORK/pkg" -czf "$TARBALL" "$APP"

say "Listo: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo "Para instalarlo en este equipo:"
echo "    tar -xzf \"$TARBALL\" -C ~ && ~/$APP/instalar.sh"
