"""
RunPod Serverless worker for Tongyi-MAI/Z-Image-Turbo.

This version wraps boot in try/except with verbose logging so any failure
shows up as a Python traceback in the worker logs (instead of just
"worker exited with exit code 1").
"""

import base64
import io
import os
import random
import sys
import traceback


def _say(msg):
    # Print to BOTH stdout and stderr so RunPod's log aggregator picks it up
    # regardless of level filter.
    print(f"[boot] {msg}", flush=True)
    print(f"[boot] {msg}", file=sys.stderr, flush=True)


_say(f"Python: {sys.version}")
_say(f"CWD: {os.getcwd()}")
_say(f"Env MODEL_ID: {os.environ.get('MODEL_ID')}")

try:
    import torch
    _say(f"torch={torch.__version__} cuda_avail={torch.cuda.is_available()} dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _say(f"dtype={DTYPE}")
except Exception as e:
    _say(f"FATAL torch import: {e}")
    traceback.print_exc()
    raise

try:
    import diffusers
    _say(f"diffusers={diffusers.__version__}")
except Exception as e:
    _say(f"FATAL diffusers import: {e}")
    traceback.print_exc()
    raise

try:
    import runpod
    _say(f"runpod={runpod.__version__ if hasattr(runpod,'__version__') else 'imported'}")
except Exception as e:
    _say(f"FATAL runpod import: {e}")
    traceback.print_exc()
    raise


MODEL_ID = os.environ.get("MODEL_ID", "Tongyi-MAI/Z-Image")
pipe = None
pipe_load_error = None


def _load_pipe():
    global pipe, pipe_load_error
    if pipe is not None or pipe_load_error is not None:
        return
    _say(f"Loading {MODEL_ID} ...")
    try:
        from diffusers import DiffusionPipeline
        pipe_local = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            trust_remote_code=True,
        )
        _say("Loaded via DiffusionPipeline. Moving to CUDA...")
        pipe_local = pipe_local.to("cuda")
        try:
            pipe_local.set_progress_bar_config(disable=True)
        except Exception:
            pass
        pipe = pipe_local
        _say("Pipeline ready on CUDA.")
    except Exception as e:
        tb = traceback.format_exc()
        pipe_load_error = f"{type(e).__name__}: {e}\n{tb}"
        _say(f"PIPELINE LOAD FAILED: {e}")
        print(tb, file=sys.stderr, flush=True)
        # Try a fallback path: maybe Z-Image needs the AutoPipeline
        try:
            _say("Trying AutoPipelineForText2Image fallback...")
            from diffusers import AutoPipelineForText2Image
            pipe_local = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID,
                torch_dtype=DTYPE,
                trust_remote_code=True,
            ).to("cuda")
            pipe = pipe_local
            pipe_load_error = None
            _say("AutoPipeline fallback worked!")
        except Exception as e2:
            _say(f"AutoPipeline fallback also failed: {e2}")
            print(traceback.format_exc(), file=sys.stderr, flush=True)


def _img_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def handler(event):
    try:
        if pipe is None:
            _load_pipe()
        if pipe_load_error:
            return {
                "error": "pipeline_load_failed",
                "detail": pipe_load_error[:3000],
                "model_id": MODEL_ID,
            }

        inp = (event or {}).get("input", {}) or {}
        prompt = inp.get("prompt") or ""
        if not prompt:
            return {"error": "missing 'prompt'"}

        negative_prompt = inp.get("negative_prompt") or None
        width = int(inp.get("width", 1024))
        height = int(inp.get("height", 1024))
        steps = int(inp.get("num_inference_steps", 50))  # full Z-Image needs ~50
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
            guidance_scale=float(inp.get("guidance_scale", 4.5)),  # full model uses real CFG
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
            "trace": traceback.format_exc()[-2500:],
        }


# Pre-load on import so the FIRST request doesn't time out on cold start.
# If this fails, the error is captured in pipe_load_error and returned by the
# handler instead of crashing the worker process.
try:
    _load_pipe()
except Exception as e:
    _say(f"Pre-load wrapper caught: {e}")
    pipe_load_error = f"pre-load wrapper: {e}\n{traceback.format_exc()}"

_say("Starting runpod.serverless.start...")
runpod.serverless.start({"handler": handler})
