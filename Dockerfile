FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hf_transfer && \
    pip install --no-cache-dir -r /app/requirements.txt

# Pre-download model into the image — slower build, much faster cold start.
ARG MODEL_ID=Tongyi-MAI/Z-Image-Turbo
ENV MODEL_ID=${MODEL_ID}
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${MODEL_ID}', allow_patterns=['*.json','*.txt','*.safetensors','*.bin','*.model','*.py'])"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
