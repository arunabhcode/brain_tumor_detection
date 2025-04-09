# Dockerfile

# Use the official PyTorch image with CUDA 11.8, CUDNN 9.
# This image tag inherently supports the Python version bundled with PyTorch 2.4, which includes 3.12[1].
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime


# --- Optional: Install additional Python packages ---
# Uncomment the following lines if you have a requirements.txt file
# Copy requirements file first to leverage Docker cache
# COPY requirements.txt .
# Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "/modules/brain_tumor/train.py"]

