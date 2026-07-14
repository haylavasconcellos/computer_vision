from PIL import Image
from .base_model import BaseInpaintingModel


class StableDiffusionInpainting(BaseInpaintingModel):
    def __init__(self, model_id: str = "runwayml/stable-diffusion-inpainting", 
                 device: str = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 **kwargs):
        super().__init__("StableDiffusionInpainting", device, **kwargs)
        self.model_id = model_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.config.update({
            "model_id": model_id,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        })

    def load_model(self) -> None:
        if self.is_loaded:
            return
        
        from diffusers import StableDiffusionInpaintPipeline
        import torch
        
        self.model = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.model = self.model.to(self.device)
        self.is_loaded = True

    def _inpaint_impl(self, image: Image.Image, mask: Image.Image, **kwargs) -> Image.Image:
        # Detectar se é restauração de foto antiga
        image_path = str(kwargs.get("image_path", "")).lower()
        
        # Usar prompt otimizado para fotos antigas se detectado
        if 'antiga' in image_path or 'vintage' in image_path or 'old' in image_path:
            default_prompt = (
                "vintage photograph restoration, seamless repair, "
                "natural skin texture, period accurate, aged paper texture, "
                "remove cracks and tears, professional photo restoration"
            )
        else:
            default_prompt = ""
        
        prompt = kwargs.get("prompt", default_prompt)
        
        # Garantir que dimensões sejam divisíveis por 8
        w, h = image.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        
        if (w, h) != (new_w, new_h):
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        
        result = self.model(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).images[0]
        
        return result
