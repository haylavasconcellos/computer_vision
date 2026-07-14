import logging
import sys
from pathlib import Path

try:
# Works in scripts
    current_file = Path(__file__).resolve()
except NameError:
    # Fallback for interactive sessions like Jupyter
    current_file = Path(sys.argv[0]).resolve() if sys.argv[0] else Path.cwd()

current_dir = current_file.parent

BASE_DATA_PATH = current_dir.parent.parent / "data"
BASE_MODEL_PATH = current_dir.parent.parent / "models"


def setup_logger(name: str = "ReconstructionPipeline", level=logging.INFO):
    """Configura logger com saída no console."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Handler de console
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Formato das mensagens
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(handler)

    return logger