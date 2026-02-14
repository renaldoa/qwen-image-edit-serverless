#!/usr/bin/env python3
"""RunPod serverless handler for Qwen-Image-Edit-2511."""

import sys
import os

# Immediate early logging before any heavy imports
print("=" * 60, flush=True)
print("Handler starting...", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"CWD: {os.getcwd()}", flush=True)
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", flush=True)
print("=" * 60, flush=True)

import base64
import io

print("Importing torch...", flush=True)
import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", flush=True)

print("Importing runpod...", flush=True)
import runpod

print("Importing PIL...", flush=True)
from PIL import Image

print("Importing diffusers...", flush=True)
from diffusers import QwenImageEditPlusPipeline
print("All imports done.", flush=True)

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

# RunPod model caching stores HF models at this path.
CACHE_DIR = "/runpod-volume/huggingface-cache/hub"
if os.path.isdir(CACHE_DIR):
    os.environ["HF_HOME"] = "/runpod-volume/huggingface-cache"
    print(f"Using RunPod model cache: {CACHE_DIR}", flush=True)
else:
    print(f"RunPod cache dir not found ({CACHE_DIR}), using default HF cache", flush=True)
    print(f"HF_HOME={os.environ.get('HF_HOME', 'not set')}", flush=True)

# Load pipeline at module level so it persists between requests
print(f"Loading {MODEL_ID} ...", flush=True)
try:
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    print("Pipeline loaded, moving to CUDA...", flush=True)
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)
    print("Model loaded and ready on CUDA.", flush=True)
except Exception as e:
    print(f"FATAL: Failed to load model: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)


def decode_image(b64_string):
    """Decode a base64 string to a PIL Image."""
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def encode_image(pil_image):
    """Encode a PIL Image to a base64 PNG string."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def handler(job):
    """Process a single image-editing request."""
    job_input = job["input"]

    # --- Parse input images ---
    if "images" in job_input:
        raw_images = job_input["images"]
        if not isinstance(raw_images, list) or len(raw_images) == 0:
            return {"error": "'images' must be a non-empty list of base64 strings."}
        images = [decode_image(b) for b in raw_images]
        multi_input = True
    elif "image" in job_input:
        images = [decode_image(job_input["image"])]
        multi_input = False
    else:
        return {"error": "Provide 'image' (base64 string) or 'images' (list of base64 strings)."}

    # --- Parse parameters ---
    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "'prompt' is required."}

    negative_prompt = job_input.get("negative_prompt", " ")
    num_inference_steps = int(job_input.get("num_inference_steps", 40))
    true_cfg_scale = float(job_input.get("true_cfg_scale", 4.0))
    seed = int(job_input.get("seed", -1))
    num_images_per_prompt = int(job_input.get("num_images_per_prompt", 1))

    generator = torch.manual_seed(seed) if seed >= 0 else None
    input_imgs = images if len(images) > 1 else images[0]

    # --- Run inference ---
    with torch.inference_mode():
        result = pipeline(
            image=input_imgs,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=1.0,
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
        )

    # --- Encode output ---
    encoded = [encode_image(img) for img in result.images]

    if num_images_per_prompt == 1 and not multi_input:
        return {"image": encoded[0]}
    return {"images": encoded}


runpod.serverless.start({"handler": handler})
