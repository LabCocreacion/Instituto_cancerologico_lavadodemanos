import sys
from pathlib import Path

# Permite importar los módulos de la raíz del proyecto (config, detector) desde las pruebas
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
