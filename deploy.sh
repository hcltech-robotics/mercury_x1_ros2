#!/bin/bash

# deploy.sh - Sync mercury_x1_ros2 workspace to remote robot via rsync

set -e  # Exit on error

# === CONFIGURATION ===
REMOTE_USER="elephant"          # Replace with your robot's username
REMOTE_HOST="192.168.10.175"   # Replace with your robot's IP or hostname
REMOTE_PATH="/home/elephant/mercury_x1_ros2"  # Destination path on robot

# Optional: SSH port (default is 22)
SSH_PORT=22

# === SCRIPT ===
echo "🚀 Deploying mercury_x1_ros2 to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

# Ensure remote parent directory exists
ssh -p $SSH_PORT "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p $(dirname "${REMOTE_PATH}")"

# Sync with rsync
rsync -avz \
  --delete \
  --exclude='.git' \
  --exclude='/build/' \
  --exclude='/install/' \
  --exclude='/log/' \
  --exclude='.vscode/' \
  --exclude='*.swp' \
  --exclude='*.swo' \
  -e "ssh -p $SSH_PORT" \
  ./ \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

echo "✅ Deployment completed!"
echo "💡 Remember to rebuild on the target: cd ~/mercury_x1_ros2 && colcon build"