#!/usr/bin/env bash
set -euo pipefail

SERVER="lucek@192.168.0.13"
REMOTE_DIR="/home/lucek/planet-hunter"

echo "==> Syncing files to ${SERVER}:${REMOTE_DIR}"
rsync -avz \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
    --exclude '.idea' --exclude 'venv' \
    --exclude 'planet_hunter.db' --exclude 'plots/' \
    --exclude 'notebooks' --exclude 'data' --exclude 'results' \
    --exclude 'to_check' --exclude '*.png' --exclude 'img.png' \
    ./ "${SERVER}:${REMOTE_DIR}/"

echo "==> Building and starting container"
ssh "$SERVER" "cd ${REMOTE_DIR} && docker compose up -d --build"

echo "==> Status"
ssh "$SERVER" "docker ps --filter name=planet-hunter"

echo "==> Done. Access at http://192.168.0.13:8420"
