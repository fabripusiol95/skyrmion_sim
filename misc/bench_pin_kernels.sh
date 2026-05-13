#!/bin/bash
# bench_pin_kernels.sh
#
# Profiles force_pin_k (naive O(N·N_pins)) vs force_pin_cll_k (cell-linked
# list O(N)) across a range of system sizes and writes a CSV summary.
#
# Usage: ./bench_pin_kernels.sh
#
# NOTE: the naive kernel is O(N × N_pins), and since N_pins ∝ area ∝ N, it
# is effectively O(N²).  For nx > MAX_NAIVE_NX the naive run is skipped to
# avoid multi-hour runtimes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$SCRIPT_DIR/build/skyrmion_sim"
IMPORTER="/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter"
BASE_CONF="$SCRIPT_DIR/inputs/test_npins.conf"
RESULTS="$SCRIPT_DIR/results/pin_scaling"
CSV="$RESULTS/summary.csv"

SIZES=(16 24 32 48 64 96 128 192 256)
MAX_NAIVE_NX=256   # naive kernel skipped above this (would take too long)

# ─── sanity checks ────────────────────────────────────────────────────────────
[[ -x "$BUILD" ]]    || { echo "ERROR: binary not found: $BUILD"; exit 1; }
[[ -x "$IMPORTER" ]] || { echo "ERROR: QdstrmImporter not found: $IMPORTER"; exit 1; }
[[ -f "$BASE_CONF" ]] || { echo "ERROR: base config not found: $BASE_CONF"; exit 1; }

mkdir -p "$RESULTS"
echo "nx,N,mode,nvtx_range,avg_us,total_us,instances" > "$CSV"

# ─── profile one (size, mode) pair ───────────────────────────────────────────
run_one() {
    local nx=$1 mode=$2 use_cells=$3
    local N=$(( nx * nx ))
    local OUTBASE="$RESULTS/N${N}_${mode}"
    local TMPCONF; TMPCONF=$(mktemp /tmp/skyrmion_XXXX.conf)

    # Patch nx, ny, and use_pin_cells in a copy of the base config
    sed \
        -e "s/^nx[[:space:]]*=.*/nx = $nx/" \
        -e "s/^ny[[:space:]]*=.*/ny = $nx/" \
        -e "s/^use_pin_cells[[:space:]]*=.*/use_pin_cells = $use_cells/" \
        "$BASE_CONF" > "$TMPCONF"

    printf "  [%s]  nx=%-4d  N=%-7d  mode=%-6s  " "$(date +%T)" "$nx" "$N" "$mode"

    nsys profile \
        --output="$OUTBASE" \
        --trace=cuda,nvtx \
        --force-overwrite=true \
        "$BUILD" "$TMPCONF" 2>&1 \
        | grep --line-buffered "\[timer\]" || true

    "$IMPORTER" "${OUTBASE}.qdstrm" 2>/dev/null

    # Parse NVTX summary → append rows to CSV
    # Columns (after stripping comma thousand-separators):
    #   1:Time(%)  2:Total_ns  3:Instances  4:Avg_ns  5:Med_ns  6:Min_ns
    #   7:Max_ns   8:StdDev    9:Style      10:Range
    nsys stats --report nvtxsum "${OUTBASE}.nsys-rep" 2>/dev/null \
        | grep "force_pin" | tr -d ',' \
        | awk -v nx="$nx" -v N="$N" -v mode="$mode" -v csv="$CSV" '
          {
              total_ns=$2; instances=$3; avg_ns=$4; range=$NF
              printf "%d,%d,%s,%s,%.2f,%.2f,%d\n",
                     nx, N, mode, range,
                     avg_ns/1000.0, total_ns/1000.0, instances >> csv
          }'

    rm -f "$TMPCONF"
}

# ─── main loop ────────────────────────────────────────────────────────────────
echo "=== Pin kernel scaling: force_pin_k vs force_pin_cll_k ==="
printf "    Sizes (nx=ny): %s\n" "${SIZES[*]}"
printf "    Naive kernel capped at nx <= %d\n\n" "$MAX_NAIVE_NX"

for nx in "${SIZES[@]}"; do
    echo "--- nx=ny=$nx ---"
    if (( nx <= MAX_NAIVE_NX )); then
        run_one "$nx" "naive" "false"
    else
        printf "  [%s]  nx=%-4d  (naive skipped — above MAX_NAIVE_NX=%d)\n" \
               "$(date +%T)" "$nx" "$MAX_NAIVE_NX"
    fi
    run_one "$nx" "cll" "true"
done

# ─── print summary ────────────────────────────────────────────────────────────
echo ""
echo "=== Results written to: $CSV ==="
echo ""
column -t -s ',' "$CSV"
