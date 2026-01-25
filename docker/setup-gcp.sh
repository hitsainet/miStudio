#!/bin/bash
# MechInterp Studio - GCP Deployment Setup Script
# URL: gcp-dkr-mistudio.hitsai.net
#
# Run this script on a fresh GCP instance with NVIDIA T4 GPU
# Prerequisites: Ubuntu 22.04 LTS with NVIDIA drivers installed

set -e

echo "========================================"
echo "MechInterp Studio GCP Deployment Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Note: Some commands may require sudo${NC}"
fi

# 1. Install Docker if not present
echo -e "\n${GREEN}[1/6] Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo -e "${GREEN}Docker installed successfully${NC}"
else
    echo -e "${GREEN}Docker already installed${NC}"
fi

# 2. Install Docker Compose plugin if not present
echo -e "\n${GREEN}[2/6] Checking Docker Compose...${NC}"
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose plugin..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
    echo -e "${GREEN}Docker Compose installed successfully${NC}"
else
    echo -e "${GREEN}Docker Compose already installed${NC}"
fi

# 3. Install NVIDIA Container Toolkit if not present
echo -e "\n${GREEN}[3/6] Checking NVIDIA Container Toolkit...${NC}"
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "Installing NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo -e "${GREEN}NVIDIA Container Toolkit installed successfully${NC}"
else
    echo -e "${GREEN}NVIDIA Container Toolkit already installed${NC}"
fi

# 4. Verify GPU is accessible from Docker
echo -e "\n${GREEN}[4/6] Verifying GPU access...${NC}"
if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU is accessible from Docker${NC}"
    docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
else
    echo -e "${RED}ERROR: GPU is not accessible from Docker${NC}"
    echo "Please ensure NVIDIA drivers are installed and nvidia-container-toolkit is configured"
    exit 1
fi

# 5. Create deployment directory
echo -e "\n${GREEN}[5/6] Setting up deployment directory...${NC}"
DEPLOY_DIR="/opt/mistudio"
sudo mkdir -p $DEPLOY_DIR
sudo chown $USER:$USER $DEPLOY_DIR

# Copy files to deployment directory
cp docker-compose.gcp.yml $DEPLOY_DIR/docker-compose.yml
cp nginx.gcp.conf $DEPLOY_DIR/nginx.gcp.conf

echo -e "${GREEN}Deployment files copied to $DEPLOY_DIR${NC}"

# 6. Pull images and start services
echo -e "\n${GREEN}[6/6] Pulling images and starting services...${NC}"
cd $DEPLOY_DIR

echo "Pulling latest images..."
docker compose pull

echo "Starting services..."
docker compose up -d

# Wait for services to be healthy
echo "Waiting for services to start..."
sleep 30

# Show status
echo -e "\n${GREEN}========================================"
echo "Deployment Complete!"
echo "========================================"
echo -e "${NC}"

echo "Service Status:"
docker compose ps

echo -e "\n${GREEN}Access URLs:${NC}"
echo "  MechInterp Studio: http://gcp-dkr-mistudio.hitsai.net"
echo "  Neuronpedia:       http://gcp-dkr-mistudio.hitsai.net:3001"
echo "  API Docs:          http://gcp-dkr-mistudio.hitsai.net/api/docs"

echo -e "\n${YELLOW}Useful Commands:${NC}"
echo "  View logs:         docker compose logs -f"
echo "  Stop services:     docker compose down"
echo "  Restart services:  docker compose restart"
echo "  Update images:     docker compose pull && docker compose up -d"

echo -e "\n${YELLOW}Note:${NC} Make sure port 80 and 3001 are open in GCP firewall rules"
