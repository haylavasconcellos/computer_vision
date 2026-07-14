from canon.config import BASE_DATA_PATH
from pathlib import Path
import cv2
from typing import Dict, Tuple


def get_kandinsky_parameters(image_type: str = "vintage") -> Dict[str, any]:
    """
    Retorna parâmetros otimizados para o Kandinsky baseado no tipo de imagem.
    
    Args:
        image_type: Tipo de imagem ('vintage', 'low_light', 'satellite', 'generic')
        
    Returns:
        Dict com 'guidance_scale', 'num_inference_steps', 'prompt', 'negative_prompt'
        
    Exemplo:
        >>> params = get_kandinsky_parameters("vintage")
        >>> result = model.inpaint(image, mask, **params)
    """
    configs = {
        "vintage": {
            "guidance_scale": 7.0,
            "num_inference_steps": 75,
            "prompt": (
                "professional photo restoration, complete only the missing areas, "
                "preserve existing facial features exactly as they are, "
                "match the vintage sepia tone perfectly, "
                "natural continuation of visible skin texture, "
                "period-appropriate clothing style from visible context, "
                "seamless repair maintaining original composition, "
                "no added objects, no decorations, no flowers, "
                "invisible restoration, faithful reproduction, "
                "match grain and lighting of surrounding area"
            ),
            "negative_prompt": (
                "added objects, extra items, flowers, decorations, jewelry, accessories, "
                "creative additions, new elements, fictional content, "
                "low quality, blurry, distorted, artifacts, inconsistent, "
                "watermark, text, logo, signature, unrealistic, "
                "obvious boundaries, visible seams, different style, different lighting"
            )
        },
        "low_light": {
            "guidance_scale": 3.5,
            "num_inference_steps": 50,
            "prompt": (
                "low light photography, dim lighting, natural shadows, "
                "ambient darkness, subtle illumination, night scene, "
                "photographic grain, authentic low light atmosphere, "
                "preserve existing lighting conditions, natural continuation"
            ),
            "negative_prompt": (
                "bright lighting, daylight, overexposed, "
                "low quality, blurry, distorted, artifacts, inconsistent, "
                "watermark, text, logo, signature, unrealistic"
            )
        },
        "satellite": {
            "guidance_scale": 4.5,
            "num_inference_steps": 50,
            "prompt": (
                "aerial satellite imagery, top-down view, earth from above, "
                "natural terrain, vegetation patterns, land formations, "
                "geographic features, consistent satellite perspective, "
                "seamless terrain continuation, natural landscape"
            ),
            "negative_prompt": (
                "ground level view, buildings, people, vehicles, "
                "low quality, blurry, distorted, artifacts, inconsistent, "
                "watermark, text, logo, signature"
            )
        },
        "generic": {
            "guidance_scale": 4.0,
            "num_inference_steps": 50,
            "prompt": (
                "natural continuation, seamless completion, "
                "consistent style, coherent image, matching context, "
                "preserve composition, no added objects"
            ),
            "negative_prompt": (
                "added objects, extra items, creative additions, "
                "low quality, blurry, distorted, artifacts, inconsistent, "
                "watermark, text, logo, signature, unrealistic"
            )
        }
    }
    
    return configs.get(image_type, configs["generic"])


def load_image_and_masks(name: str):
    base = BASE_DATA_PATH / "T4"
    print("Reading from", base)
    img_path = base / "imagens" / f"{name}.jpg"
    if not img_path.exists():
        img_path = base / "imagens" / f"{name}.png"

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    masks = []
    for i in range(1, 4):
        mask_path = base / "mascaras" / f"{name}_mask_{i}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        masks.append(mask)

    return image, masks

def get_image_path(name: str):
    base = BASE_DATA_PATH / "T4"
    img_path = base / "imagens" / f"{name}.jpg"
    
    if not img_path.exists():
        img_path = base / "imagens" / f"{name}.png"

    return img_path

def apply_mask(image, mask):
    # Invert mask so white (to remove) becomes 0, black (keep) becomes 255
    inverse_mask = cv2.bitwise_not(mask)

    # Apply mask to image
    result = cv2.bitwise_and(image, image, mask=inverse_mask)
    return result