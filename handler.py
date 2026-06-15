"""
RunPod Serverless worker for Tongyi-MAI/Z-Image-Turbo.

Input schema:
{
  "input": {
    "prompt": "...",
    "negative_prompt": "",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 8,
    "seed": 12345,
    "num_images": 1
  }
}

Output:
{
  "images": ["data:image/png;base64,..."],
  "width": 1024,
  "height": 1024,
  "seed_used": 12345
}
"""

import base64
import io
import os
import random
import traceback

import runpod
import torch
from diffusers import DiffusionPipeline

MODEL_ID = os.environ.get("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print(f"[boot] Loading {MODEL_ID} (dtype={DTYPE})...")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True,
)
pipe = pipe.to("cuda")
try:
    pipe.set_progress_bar_config(disable=True)
except Exception:
    pass
print("[boot] Model ready.")


def _img_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def handler(event):
    try:
        inp = (event or {}).get("input", {}) or {}
        prompt = inp.get("prompt") or ""
        if not prompt:
            return {"error": "missing 'prompt'"}

        negative_prompt = inp.get("negative_prompt", "") or None
        width = int(inp.get("width", 1024))
        height = int(inp.get("height", 1024))
        steps = int(inp.get("num_inference_steps", 8))
        num_images = max(1, min(int(inp.get("num_images", 1)), 4))
        seed = inp.get("seed")
        if seed is None:
            seed = random.randint(0, 2**31 - 1)
        seed = int(seed)

        gen = torch.Generator(device="cuda").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            num_images_per_prompt=num_images,
            generator=gen,
        )
        images = [_img_to_data_url(im) for im in result.images]

        return {
            "images": images,
            "width": width,
            "height": height,
            "seed_used": seed,
            "model": MODEL_ID,
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc()[-2000:],
        }


runpod.serverless.start({"handler": handler})
