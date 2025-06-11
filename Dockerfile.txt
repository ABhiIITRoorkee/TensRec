# Use the official PyTorch image with CUDA 12.1 and cuDNN8 support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . /app

# Install everything except torch itself
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

CMD ["python3", "main.py"]
