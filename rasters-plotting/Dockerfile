# Use Python 3.11 base image
FROM python:3.11-slim

# Set the environment variable to prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages (with cleanup)
RUN apt-get update && apt-get install -y time && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to install dependencies
COPY requirements.txt /app/requirements.txt

# Set working directory
WORKDIR /app

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Set up environment variable for PRP setup
ENV ENDPOINT_URL="https://s3.braingeneers.gi.ucsc.edu"
