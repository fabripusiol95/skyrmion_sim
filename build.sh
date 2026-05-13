#!/bin/bash
set -e

# ── Modules ───────────────────────────────────────────────────────────────────
# CUDA 11.4 supports GCC up to 10; gcc/11.1.0 is incompatible.
module load gcc/8.2.0
module load cuda/11.4.0
module load cmake-3.19.2-gcc-8.2.0

# ── GPU architecture (auto-detected from all available GPUs) ─────────────────
# CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader \
#     | tr -d '.' | sort -u | paste -sd ';')
# echo "Detected CUDA archs: $CUDA_ARCH"

CUDA_ARCH="70;75;80;86" # --- IGNORE --- (for testing on a machine with only one GPU)

# ── Build ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

make -j"$(nproc)"

echo ""
echo "Build complete: $BUILD_DIR/skyrmion_sim"
