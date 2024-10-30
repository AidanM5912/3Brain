FROM python:3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y time

# RUN python -m pip install --force-reinstall 'git+https://github.com/SpikeInterface/spikeinterface.git'
RUN pip install "spikeinterface==0.98.0"
# RUN python -m pip install --force-reinstall 'git+https://github.com/braingeneers/braingeneerspy.git#egg=braingeneerspy[iot, data]'
RUN pip install "braingeneers[iot, analysis, data]"

RUN pip install numpy \
    deprecated \
    numba \
    tbb \
    h5py \
    scipy \
    glances \
    pynwb

# COPY docker/requirements.txt ./src/requirements.txt
# RUN set -ex && \
#     pip install -r ./src/requirements.txt

# PRP setup
ENV ENDPOINT_URL="https://braingeneers.gi.ucsc.edu"

WORKDIR /app

COPY . /app


# qm_curation:v0.2
