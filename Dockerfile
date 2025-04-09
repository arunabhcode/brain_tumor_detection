# Dockerfile

# Use the official PyTorch image with CUDA 11.8, CUDNN 9.
# This image tag inherently supports the Python version bundled with PyTorch 2.4, which includes 3.12[1].
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg

COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/modules/brain_tumor/train.py"]

