FROM runpod/pytorch:2.5.1-py3.10-cuda12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "transformers>=4.51.3" \
        "accelerate>=0.26.0" \
        "git+https://github.com/huggingface/diffusers" \
        "runpod>=1.6.0" \
        "Pillow>=10.0.0"

COPY handler.py /handler.py
WORKDIR /

CMD ["python", "-u", "/handler.py"]
