#!/usr/bin/env bash
set -euo pipefail

EXECUTABLE="./build/skyrmion_sim"
BASE_CONF="inputs/depinning_test.conf"
TMP_CONF=$(mktemp /tmp/skyrmion_XXXXXX.conf)

trap 'rm -f "$TMP_CONF"' EXIT

mkdir -p ../results/depinning

for FD in $(seq -f "%.4f" 0 0.0005 0.001); do
    TAG="Force_F_D_${FD}"
    echo "Running F_D = ${FD}  →  run_tag = ${TAG}"

    sed \
        -e "s|^F_D\s*=.*|F_D         = ${FD}|" \
        -e "s|^output_dir\s*=.*|output_dir  = results/depinning|" \
        -e "s|^run_tag\s*=.*|run_tag     = ${TAG}|" \
        "$BASE_CONF" > "$TMP_CONF"

    "$EXECUTABLE" "$TMP_CONF"
done

echo "All runs complete. Results in results/depinning/"
