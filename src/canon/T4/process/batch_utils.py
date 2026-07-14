import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import json
from canon.T4 import get_model, list_available_models


def save_results(results: Dict[str, Any], output_dir: Path, prefix: str = "result"):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, result in results.items():
        if "error" in result:
            print(f"Skipping {model_name} due to error: {result['error']}")
            continue
        
        image_path = output_dir / f"{prefix}_{model_name}.png"
        result["image"].save(image_path)
        
        metadata = {
            "model_name": result["model_name"],
            "inference_time": result["inference_time"],
            "device": result["device"],
            "config": result["config"]
        }
        
        meta_path = output_dir / f"{prefix}_{model_name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {model_name} result to {image_path}")


def batch_inpaint_single_image(
    image_path: Path,
    mask_paths: List[Path],
    models: List[str],
    output_dir: Path,
    device: str = "cuda",
    **model_kwargs
):
    image = Image.open(image_path).convert("RGB")
    image_name = image_path.stem
    
    for mask_idx, mask_path in enumerate(mask_paths):
        mask = Image.open(mask_path).convert("L")
        mask_name = mask_path.stem
        
        print(f"\nProcessing {image_name} with {mask_name}")
        print("-" * 60)
        
        for model_name in models:
            try:
                print(f"  Running {model_name}...")
                model = get_model(model_name, device=device, **model_kwargs)
                result = model.inpaint(image, mask, image_path=image_path)
                
                prefix = f"{image_name}_{mask_name}"
                save_results(
                    {model_name: result},
                    output_dir / image_name / mask_name,
                    prefix
                )
                
                print(f"    Time: {result['inference_time']:.2f}s")
                model.unload_model()
                
            except Exception as e:
                print(f"    Error: {e}")


def batch_inpaint_directory(
    images_dir: Path,
    masks_dir: Path,
    models: List[str],
    output_dir: Path,
    device: str = "cuda",
    image_extensions: List[str] = [".jpg", ".jpeg", ".png"],
    **model_kwargs
):
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    for image_path in sorted(image_files):
        image_name = image_path.stem
        
        mask_patterns = [
            f"{image_name}_mask_*.png",
            f"{image_name}_*.png",
        ]
        
        mask_paths = []
        for pattern in mask_patterns:
            mask_paths.extend(masks_dir.glob(pattern))
        
        if not mask_paths:
            print(f"No masks found for {image_name}, skipping")
            continue
        
        print(f"\nFound {len(mask_paths)} masks for {image_name}")
        
        batch_inpaint_single_image(
            image_path,
            mask_paths,
            models,
            output_dir,
            device,
            **model_kwargs
        )


def compare_models_grid(
    image_path: Path,
    mask_path: Path,
    models: List[str],
    output_path: Path,
    device: str = "cuda",
    **model_kwargs
):
    import matplotlib.pyplot as plt
    
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    results = {}
    for model_name in models:
        try:
            print(f"Running {model_name}...")
            model = get_model(model_name, device=device, **model_kwargs)
            result = model.inpaint(image, mask, image_path=image_path)
            results[model_name] = result
            model.unload_model()
            print(f"  Time: {result['inference_time']:.2f}s")
        except Exception as e:
            print(f"  Error: {e}")
    
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models + 2, figsize=(5 * (num_models + 2), 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')
    
    for idx, (model_name, result) in enumerate(results.items(), start=2):
        axes[idx].imshow(result["image"])
        axes[idx].set_title(f"{model_name}\n{result['inference_time']:.2f}s")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch inpainting utilities")
    parser.add_argument("--mode", choices=["single", "batch", "compare"], required=True)
    parser.add_argument("--image", type=Path, help="Input image path")
    parser.add_argument("--mask", type=Path, help="Input mask path")
    parser.add_argument("--images-dir", type=Path, help="Directory with images")
    parser.add_argument("--masks-dir", type=Path, help="Directory with masks")
    parser.add_argument("--output", type=Path, required=True, help="Output directory/file")
    parser.add_argument("--models", nargs="+", help="Models to use")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    
    args = parser.parse_args()
    
    if args.models is None:
        args.models = list_available_models()
    
    if args.mode == "single":
        if not args.image or not args.mask:
            print("Error: --image and --mask required for single mode")
            sys.exit(1)
        
        batch_inpaint_single_image(
            args.image,
            [args.mask],
            args.models,
            args.output,
            args.device,
            num_inference_steps=args.steps
        )
    
    elif args.mode == "batch":
        if not args.images_dir or not args.masks_dir:
            print("Error: --images-dir and --masks-dir required for batch mode")
            sys.exit(1)
        
        batch_inpaint_directory(
            args.images_dir,
            args.masks_dir,
            args.models,
            args.output,
            args.device,
            num_inference_steps=args.steps
        )
    
    elif args.mode == "compare":
        if not args.image or not args.mask:
            print("Error: --image and --mask required for compare mode")
            sys.exit(1)
        
        compare_models_grid(
            args.image,
            args.mask,
            args.models,
            args.output,
            args.device,
            num_inference_steps=args.steps
        )
