#!/bin/bash

# This script builds the Docker image for the project.
# It uses the Dockerfile located in the current directory and tags the image with the name "myapp".
# Usage: ./build.sh
# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found. Please install Docker to proceed."
    exit 1
fi
# Check if Docker is running
if ! docker info &> /dev/null
then
    echo "Docker is not running. Please start Docker to proceed."
    exit 1
fi
# Build the Docker image
docker build -t dl/brain_tumor .