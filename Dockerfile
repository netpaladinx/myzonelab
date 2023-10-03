# 21.06-py3:
#   CUDA 11.3, PyTorch 1.9, TensorRT 7.2.3
# 22.07-py3:
#   CUDA 11.7, PyTorch 1.13, TensorRT 8.4.1

# FROM nvcr.io/nvidia/pytorch:22.07-py3
FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0

RUN pip install --upgrade pip \
    & pip install --user \
    pyyaml \
    pillow \
    thop \
    numpy \
    pandas \
    matplotlib \
    scipy \
    scikit-image \
    scikit-learn \
    seaborn \
    opencv-python \
    opencv-python-headless \
    opencv-contrib-python \
    onnx \
    onnxruntime \
    albumentations \
    tensorboard \
    jupyterlab \
    plotly \
    kaleido

WORKDIR /workspace

ENTRYPOINT /workspace/startup.sh