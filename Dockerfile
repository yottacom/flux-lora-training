# Use the NVIDIA CUDA base image
# FROM nvidia/cuda:12.0.1-runtime-ubuntu22.04
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Install apt packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libomp-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    libgl1-mesa-glx \
    python3-pip \
    gdb \
    valgrind \
    git

# Update pip to the latest version
RUN python3 -m pip install --upgrade pip

# Copy all files into the /app directory
COPY . /app

# Set the working directory
WORKDIR /app

# Install Python packages
RUN python3 -m pip install -r requirements.txt

