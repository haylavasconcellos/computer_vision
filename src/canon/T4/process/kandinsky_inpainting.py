from PIL import Image
from .base_model import BaseInpaintingModel


class KandinskyInpainting(BaseInpaintingModel):
    def __init__(self, 
                 model_id: str = "kandinsky-community/kandinsky-2-2-decoder-inpaint",
                 prior_id: str = "kandinsky-community/kandinsky-2-2-prior",
                 device: str = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 4.0,
                 **kwargs):
        super().__init__("KandinskyInpainting", device, **kwargs)
        self.model_id = model_id
        self.prior_id = prior_id
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.config.update({
            "model_id": model_id,
            "prior_id": prior_id,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale
        })

    def load_model(self) -> None:
        if self.is_loaded:
            return
        
        from diffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline
        import torch
        
        self.prior = KandinskyV22PriorPipeline.from_pretrained(
            self.prior_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.prior = self.prior.to(self.device)
        
        self.model = KandinskyV22InpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model = self.model.to(self.device)
        self.is_loaded = True

    def _inpaint_impl(self, image: Image.Image, mask: Image.Image, **kwargs) -> Image.Image:
        # Detecção automática do tipo de imagem baseado no nome do arquivo (se fornecido)
        image_path = str(kwargs.get("image_path", "")).lower()
        
        # Detectar tipo e definir prompt apropriado
        if 'antiga' in image_path or 'vintage' in image_path or 'old' in image_path or 'restored' in image_path:
            # Prompt otimizado para fotos antigas - foco em PRESERVAÇÃO FIEL
            default_prompt = (
                "professional photo restoration, complete only the missing areas, "
                "preserve existing facial features exactly as they are, "
                "match the vintage sepia tone perfectly, "
                "natural continuation of visible skin texture, "
                "period-appropriate clothing style from visible context, "
                "seamless repair maintaining original composition, "
                "no added objects, no decorations, no flowers, "
                "invisible restoration, faithful reproduction, "
                "match grain and lighting of surrounding area"
            )
            default_guidance = 7.0  # Menor autonomia do modelo
            default_steps = 75  # Mais steps para melhor qualidade
        elif 'baixa_luz' in image_path or 'low_light' in image_path or 'noturna' in image_path:
            default_prompt = (
                "low light photography, dim lighting, natural shadows, "
                "ambient darkness, subtle illumination, night scene, "
                "photographic grain, authentic low light atmosphere, "
                "preserve existing lighting conditions, natural continuation"
            )
            default_guidance = 3.5
            default_steps = 50
        elif 'satelite' in image_path or 'satellite' in image_path or 'aerial' in image_path:
            default_prompt = (
                "aerial satellite imagery, top-down view, earth from above, "
                "natural terrain, vegetation patterns, land formations, "
                "geographic features, consistent satellite perspective, "
                "seamless terrain continuation, natural landscape"
            )
            default_guidance = 4.5
            default_steps = 50
        else:
            # Genérico - para uso quando não há informação sobre o tipo
            default_prompt = (
                "natural continuation, seamless completion, "
                "consistent style, coherent image, matching context, "
                "preserve composition, no added objects"
            )
            default_guidance = 4.0
            default_steps = 50
        
        prompt = kwargs.get("prompt", default_prompt)
        guidance_scale = kwargs.get("guidance_scale", default_guidance)
        num_inference_steps = kwargs.get("num_inference_steps", default_steps)
        
        # Negative prompt mais agressivo para evitar adições indesejadas
        negative_prompt = kwargs.get(
            "negative_prompt", 
            "added objects, extra items, flowers, decorations, jewelry, accessories, "
            "creative additions, new elements, fictional content, "
            "low quality, blurry, distorted, artifacts, inconsistent, "
            "watermark, text, logo, signature, unrealistic, "
            "obvious boundaries, visible seams, different style, different lighting"
        )
        
        # Kandinsky requer múltiplos de 64 (não 8!)
        w, h = image.size
        new_w = (w // 64) * 64
        new_h = (h // 64) * 64
        
        # Garantir tamanho mínimo de 512
        if new_w < 512:
            new_w = 512
        if new_h < 512:
            new_h = 512
        
        if (w, h) != (new_w, new_h):
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        
        print(f"\n{prompt=}\n")
        image_embeds, negative_image_embeds = self.prior(
            prompt=prompt,
            negative_prompt=negative_prompt
        ).to_tuple()
        
        result = self.model(
            image=image,
            mask_image=mask,
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            num_inference_steps=num_inference_steps,  # Usa steps ajustados por tipo
            guidance_scale=guidance_scale  # Usa o guidance ajustado por tipo
        ).images[0]
        
        return result

    def unload_model(self) -> None:
        import torch
        
        if self.model is not None:
            del self.model
            self.model = None
        
        if hasattr(self, 'prior') and self.prior is not None:
            del self.prior
            self.prior = None
        
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
