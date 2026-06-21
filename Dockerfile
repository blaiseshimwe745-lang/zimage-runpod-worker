FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/root/.cache/huggingface \
    PIP_BREAK_SYSTEM_PACKAGES=1

# ---- Python 3.12 native on Ubuntu 24.04 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# ---- PyTorch with Blackwell (sm_120) support via CUDA 12.8 ----
# Falls back to nightly if 2.7 stable isn't published yet.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision \
    || pip install --no-cache-dir --pre \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        torch torchvision

# ---- runpod + HF stack ----
RUN pip install --no-cache-dir \
        runpod hf_transfer \
        "transformers>=4.46" "accelerate>=1.1" "safetensors>=0.4.5" \
        sentencepiece protobuf Pillow numpy \
        git+https://github.com/huggingface/diffusers

# ---- pre-download model into image (slow build, fast cold start) ----
ARG MODEL_ID=Tongyi-MAI/Z-Image
ENV MODEL_ID=${MODEL_ID}
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', allow_patterns=['*.json','*.txt','*.safetensors','*.bin','*.model','*.py'])"

WORKDIR /app
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
