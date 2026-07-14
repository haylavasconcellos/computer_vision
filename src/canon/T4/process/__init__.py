from .base_model import BaseInpaintingModel
from .stable_diffusion_inpainting import StableDiffusionInpainting
from .paint_by_example import PaintByExample
from .kandinsky_inpainting import KandinskyInpainting
from .model_registry import (
    get_model, 
    list_available_models, 
    check_model_availability,
    run_all_inpainting_models
)
from .photo_restoration_utils import (
    detect_damage_mask,
    preprocess_old_photo,
    restore_photo,
    create_manual_mask_interactive
)

__all__ = [
    # Modelos de inpainting
    "BaseInpaintingModel",
    "StableDiffusionInpainting",
    "PaintByExample",
    "KandinskyInpainting",
    # Registry de modelos
    "get_model",
    "list_available_models",
    "check_model_availability",
    "run_all_inpainting_models",
    # Utilitários de restauração de fotos
    "detect_damage_mask",
    "preprocess_old_photo",
    "restore_photo",
    "create_manual_mask_interactive",
]
