#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE_NAME="cvevolve:latest"

# --- Resource limits (adjust to your machine) ---
MEMORY_LIMIT="${MEMORY_LIMIT:-6g}"
CPU_LIMIT="${CPU_LIMIT:-4}"

# --- Build if image doesn't exist ---
if ! podman image exists "$IMAGE_NAME"; then
    echo "Building cvevolve image..."
    podman build -t "$IMAGE_NAME" "$PROJECT_DIR/CVEvolve"
fi

# --- Load API key ---
# Reads from ~/.argo_api_key so the secret stays off OneDrive.
# Create it once:  echo 'your-key-here' > ~/.argo_api_key && chmod 600 ~/.argo_api_key
KEY_FILE="${HOME}/.argo_api_key"
if [ -n "${ARGO_API_KEY:-}" ]; then
    : # already set in environment, use it
elif [ -f "$KEY_FILE" ]; then
    ARGO_API_KEY="$(cat "$KEY_FILE" | tr -d '[:space:]')"
else
    echo "Error: ARGO_API_KEY not set and $KEY_FILE not found."
    echo "Create it:  echo 'your-key' > $KEY_FILE && chmod 600 $KEY_FILE"
    exit 1
fi

echo "Starting CVEvolve 5x5 binned run"
echo "  Memory limit: $MEMORY_LIMIT"
echo "  CPU limit:    $CPU_LIMIT"
echo "  Sessions at:  $SCRIPT_DIR/sessions"
echo ""

RAW_SCANS_HOST="$PROJECT_DIR/raw_scans"
RAW_SCANS_CONTAINER="/mnt/c/Users/trobson/OneDrive - Argonne National Laboratory/2026-1_Luo/raw_scans"

if [ ! -d "$RAW_SCANS_HOST" ]; then
    echo "Warning: raw_scans directory not found at $RAW_SCANS_HOST"
    echo "H5 file access inside the container will fail."
fi

podman run --rm \
    -e ARGO_API_KEY="$ARGO_API_KEY" \
    -v "$SCRIPT_DIR:/data:z" \
    -v "$RAW_SCANS_HOST:$RAW_SCANS_CONTAINER:ro,z" \
    --memory "$MEMORY_LIMIT" \
    --cpus "$CPU_LIMIT" \
    "$IMAGE_NAME" \
    cvevolve run \
        --config /data/config.container.yaml \
        --prompt /data/prompt.md \
        --holdout-test-prompt /data/holdout_test_prompt.md \
        --enable-logging
