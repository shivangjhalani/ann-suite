#!/bin/bash
set -e

# Setup script for a fresh Ubuntu 22.04 VM
# Installs system dependencies, Docker, uv, and sets up the ann-suite project.



echo "Starting setup..."

# 1. Update and Upgrade System
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# 2. Install Basic Dependencies
echo "Installing git, curl, wget, and build essentials..."
sudo apt-get install -y git curl wget build-essential ca-certificates gnupg

# 3. Install Docker (Official Method)
echo "Installing Docker..."
# Add Docker's official GPG key:
sudo install -m 0755 -d /etc/apt/keyrings
if [ ! -f /etc/apt/keyrings/docker.asc ]; then
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
fi

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add current user to docker group to run docker without sudo
sudo usermod -aG docker "$USER"
echo "Docker installed. User added to docker group (requires re-login to take effect)."

# 4. Install uv (Fast Python Package Installer)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for the current session
# The installer typically installs to ~/.local/bin or ~/.cargo/bin
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Persist PATH in .bashrc for future sessions
if [ -f "$HOME/.bashrc" ]; then
    if ! grep -q 'export PATH="$HOME/.local/bin' "$HOME/.bashrc"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        echo "Added ~/.local/bin to PATH in ~/.bashrc"
    fi
fi

if ! command -v uv &> /dev/null; then
    echo "Error: uv could not be found after installation. Please check the install logs."
    exit 1
fi
echo "uv installed successfully version: $(uv --version)"

# 5. Clone Repository
REPO_URL="https://github.com/shivangjhalani/ann-suite.git"
DIR_NAME="ann-suite"

if [ -d "$DIR_NAME" ]; then
    echo "Directory $DIR_NAME already exists. Pulling latest changes..."
    cd "$DIR_NAME"
    git pull
else
    echo "Cloning $REPO_URL..."
    git clone "$REPO_URL"
    cd "$DIR_NAME"
fi

# 6. Install Project Dependencies
echo "Installing project dependencies with uv..."
# 'uv sync' respects the lockfile and installs usage dependencies
uv sync

echo "==========================================="
echo "Setup Complete!"
echo "1. Dependencies installed."
echo "2. Repo cloned to $(pwd)."
echo "3. Python environment set up via uv."
echo ""
echo "IMPORTANT: 'uv' is installed in ~/.local/bin."
echo "To use 'uv' in this current terminal, you MUST run:"
echo "    source ~/.bashrc"
echo "    # OR"
echo "    export PATH=\$HOME/.local/bin:\$PATH"
echo ""
echo "NOTE: To use Docker without sudo, please log out and log back in, or run: 'newgrp docker'"
echo "To activate the virtual environment: 'source .venv/bin/activate'"
echo "==========================================="
