from abc import ABC, abstractmethod
import time
from typing import Union, Dict, Any, Tuple
from PIL import Image
import numpy as np
import torch


class BaseInpaintingModel(ABC):
    def __init__(self, model_name: str, device: str = None, **kwargs):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.config = kwargs
        self.is_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def _inpaint_impl(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        pass

    def preprocess(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray]) -> Tuple[Image.Image, Image.Image]:
        if isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                if len(image.shape) == 2:
                    image = Image.fromarray(image).convert("RGB")
                else:
                    image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))
        
        if isinstance(mask, np.ndarray):
            if mask.dtype == np.uint8:
                mask = Image.fromarray(mask).convert("L")
            else:
                mask = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        
        if mask.mode != "L":
            mask = mask.convert("L")
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image, mask

    def postprocess(self, result: Image.Image) -> Image.Image:
        return result

    def inpaint(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray], **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_model()
        
        image_pil, mask_pil = self.preprocess(image, mask)
        
        start_time = time.time()
        result = self._inpaint_impl(image_pil, mask_pil, **kwargs)
        inference_time = time.time() - start_time
        
        result = self.postprocess(result)
        
        return {
            "image": result,
            "inference_time": inference_time,
            "model_name": self.model_name,
            "device": self.device,
            "config": self.config
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "config": self.config
        }

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
