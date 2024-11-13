FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y time

RUN pip install h5py \
neuroconv \ 
spikeinterface \ 
braingeneers \
python-dateutil \ 
memory-profiler

#PRP setup
ENV ENDPOINT_URL="https://s3.braingeneers.gi.ucsc.edu"

COPY . /src
WORKDIR /src