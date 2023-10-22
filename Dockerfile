#
# DOCKER_BUILDKIT=0 docker build -t opensetdetector:v0.0.2 .
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install --no-install-recommends -y \
    wget git ffmpeg vim nano python3-pip && \
    apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove && \
    rm -rf /var/cache/apt/archives/

### APT END ###
RUN python3 -m pip install --upgrade --no-cache-dir pip setuptools wheel gradio

## GroundingDINO
ENV CUDA_HOME=/usr/local/cuda-11.7/
ENV PATH=/usr/local/cuda-11.7/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
ENV BUILD_WITH_CUDA=True

RUN mkdir /develop
RUN cd /develop
RUN git clone --recurse-submodules -j4 https://github.com/mervo/GroundingDINO.git . \
    && pip install --no-cache-dir ./video_utils \
    && pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install --no-cache-dir -e .

WORKDIR /workspace
