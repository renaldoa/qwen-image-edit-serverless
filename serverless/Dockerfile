FROM runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2204

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
