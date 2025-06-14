#!/bin/bash

# Navigate to the Docker directory from the script location
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_ROOT/docker"

echo "Changing to Docker directory: $DOCKER_DIR"
cd "$DOCKER_DIR" || { echo "Failed to change to Docker directory"; exit 1; }

echo "Building and starting Docker Compose services..."
docker compose -f docker-compose.yml up -d || { echo "Failed to start Docker Compose services"; exit 1; }

echo "Services started successfully!"
echo "Triton server is accessible at: http://localhost:8000"
echo "Postgres is accessible at: localhost:5432"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop services: docker-compose down"
