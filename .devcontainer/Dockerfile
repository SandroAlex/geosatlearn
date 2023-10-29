# Base image.
FROM ubuntu:20.04

# User specification.
ARG USER_NAME=alex
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Disable apt from prompting.
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies.
RUN apt-get update --yes --no-install-recommends && \
	apt-get install --yes --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    apt-utils \
    dialog \
    git \
    vim \
	sudo \
    software-properties-common \
    libgtk2.0-dev \
    pkg-config \
    ffmpeg \
    libsm6 \
    libxext6 \
    python-is-python3

# Add the official Deadsnakes PPA to your system's repository list.
RUN add-apt-repository ppa:deadsnakes/ppa --yes && \
    apt update --yes

# Newer python version.
RUN apt install --yes \
    python3.10 \
    python3.10-venv \
    python3.10-dev

# Clean.
RUN apt-get autoremove --yes && \
	apt-get clean --yes

# Create a system user using user variables.
RUN groupadd --gid $USER_GID $USER_NAME && \
	useradd --uid $USER_UID --gid $USER_GID --create-home $USER_NAME

# Use apt in interactive mode when we are actually using docker container.
ENV DEBIAN_FRONTEND=dialog

# Add the non-root user to the sudo group and grant them sudo privileges.
# No password for sudo commands.
RUN adduser $USER_NAME sudo && \
    echo "%sudo ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set the newly created system user as default, instead of root.
USER $USER_NAME

# Default shell inside container.
ENV SHELL=/bin/bash

# Install GDAL requirements.
# https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html
RUN sudo add-apt-repository ppa:ubuntugis/ppa --yes && \
    sudo apt-get update --yes && \
    sudo apt-get install --yes gdal-bin libgdal-dev
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# There are a handful of environment variables that must be set in order for GDAL 
# to locate and read the raster files correctly. 
ENV CPL_VSIL_CURL_USE_HEAD=FALSE
ENV GDAL_DISABLE_READDIR_ON_OPEN=YES
ENV GDAL_HTTP_COOKIEJAR=/tmp/cookies.txt
ENV GDAL_HTTP_COOKIEFILE=/tmp/cookies.txt