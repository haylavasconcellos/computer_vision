from PIL import Image
from .base_model import BaseInpaintingModel


class PaintByExample(BaseInpaintingModel):
    def __init__(self, 
                 model_id: str = "Fantasy-Studio/Paint-by-Example",
                 device: str = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 5.0,
                 **kwargs):
        super().__init__("PaintByExample", device, **kwargs)
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
        
        from diffusers import DiffusionPipeline
        import torch
        
        self.model = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        self.model = self.model.to(self.device)
        self.is_loaded = True

    def _inpaint_impl(self, image: Image.Image, mask: Image.Image, **kwargs) -> Image.Image:
        # Paint-by-Example precisa de uma imagem de exemplo para guiar o inpainting
        # Por padrão, usa a própria imagem (não ideal, mas funciona)
        # Para melhores resultados, passe example_image=<outra_imagem_similar>
        example_image = kwargs.get("example_image", image)
        
        # Garantir que dimensões sejam divisíveis por 8
        w, h = image.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        
        if (w, h) != (new_w, new_h):
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
            if example_image is not image:
                example_image = example_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                example_image = image
        
        # Converter máscara para escala de cinza
        mask_gray = mask.convert('L')
        
        result = self.model(
            example_image=example_image,
            image=image,
            mask_image=mask_gray,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).images[0]
        
        return result
