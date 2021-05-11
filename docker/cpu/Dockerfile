FROM continuumio/anaconda3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV SHELL=/bin/bash

ARG PROJECT=sample_embed_2d

# Install packages
RUN apt-get update && apt-get install -y \
  nano \
  htop \
  libsndfile1 \
  wget \
  unzip \
  ffmpeg

# Upgrade pip
RUN pip install --upgrade pip

# Create working directory
RUN mkdir /$PROJECT

# Install python packages
ADD requirements.txt /$PROJECT
WORKDIR /$PROJECT
RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
