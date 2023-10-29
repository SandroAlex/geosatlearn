#!/bin/bash
set -ex

# Create virtual environment.
VENV_NAME=venv
python3.10 -m venv ${VENV_NAME}

# Activate it.
source ./${VENV_NAME}/bin/activate

# Upgrade pip.
pip install --upgrade pip

# Install all python dependencies.
pip install --upgrade -r /geosatlearn_app/requirements.txt

# Install GDAL.
pip install GDAL==$(gdal-config --version)

# Install development tools.
pip install \
    ipython \
    jupyterlab

# Authentication with Earthdata and the .netrc file.
source .env
echo "machine urs.earthdata.nasa.gov login ${NASA_USERNAME} password ${NASA_PASSWORD}" > ~/.netrc