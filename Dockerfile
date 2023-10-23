# DOCKER_BUILDKIT=0 docker build -t text_object_detection .

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

ENV cwd="/workspace/"
WORKDIR $cwd

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ENV TORCH_CUDA_ARCH_LIST="7.5 8.6"

RUN apt-get -y update \
    && apt-get -y upgrade

RUN apt-get install --no-install-recommends -y --fix-missing \
    software-properties-common \
    build-essential \
    libgl1-mesa-glx \
    git ffmpeg vim nano

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

RUN rm -rf /var/cache/apt/archives/

### APT END ###
RUN apt-get update && apt-get install -y python3-pip
RUN python3 -m pip install --upgrade pip setuptools

## GroundingDINO
RUN git clone --recurse-submodules -j4 https://github.com/mervo/GroundingDINO.git . \
    && pip install ./video_utils \
    && pip install -r requirements.txt

ENV CUDA_HOME=/usr/local/cuda-12.1/
ENV PATH=/usr/local/cuda-12.1/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
ENV BUILD_WITH_CUDA=True

RUN pip install -e .

# wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
ADD weights weights
# https://huggingface.co/bert-base-uncased/tree/main
# /home/user/.cache/huggingface/hub/models--bert-base-uncased

RUN mkdir -p /root/.cache/huggingface/hub/models--bert-base-uncased && cp -r weights/huggingface/hub/models--bert-base-uncased /root/.cache/huggingface/hub/