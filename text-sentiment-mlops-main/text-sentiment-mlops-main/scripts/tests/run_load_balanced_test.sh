#!/bin/bash

# Stop any existing containers
echo "Stopping existing containers..."
docker-compose -f ../../deployment/docker-compose.yml down

# Start the load-balanced services
echo "Starting load-balanced services..."
docker-compose -f ../../deployment/docker-compose.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Run the load balancing test
echo "Running load balancing test..."
python test_load_balancing.py

# Show logs from all instances
echo -e "\n=== Service Logs ==="
docker-compose -f ../../deployment/docker-compose.yml logs --tail=50

# Stop the services
echo -e "\nStopping services..."
docker-compose -f ../../deployment/docker-compose.yml down 