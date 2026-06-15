"""Quick local-only sanity check (not used in RunPod). Invokes handler() directly."""
from handler import handler

out = handler({
    "input": {
        "prompt": "A cinematic photo of a tiny astronaut sitting on a moss-covered rock, dramatic light, photorealistic",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 8,
        "num_images": 1,
    }
})
print("keys:", list(out.keys()))
print("error:", out.get("error"))
print("len(images):", len(out.get("images", [])))
