#!/usr/bin/env bash
# Benchmark main-loop timing vs system size (nx=ny, use_pin_cells=false).
# Sizes: 16,24,..., |  5 runs each  |  parses [timer] line from stdout.
set -e

EXECUTABLE="./build/skyrmion_sim"
BASE_CONF="inputs/test_npins.conf"
OUT_FILE="results/bench_scaling_true.txt"
N_RUNS=5

TMP_CONF=$(mktemp /tmp/skyrmion_bench_XXXXXX.conf)
trap 'rm -f "$TMP_CONF"' EXIT

mkdir -p results

printf "%-6s  %-14s  %-14s\n" "nx" "total_ms" "ms_per_step" > "$OUT_FILE"

for nx in 10 16 24 32 48 64 96 128 192 256 384 512 768; do
    sed \
        -e "s|^\s*nx\s*=.*|nx          = ${nx}|" \
        -e "s|^\s*ny\s*=.*|ny          = ${nx}|" \
        -e "s|^\s*use_pin_cells\s*=.*|use_pin_cells = true|" \
        -e "s|^\s*verbose\s*=.*|verbose     = true|" \
        "$BASE_CONF" > "$TMP_CONF"

    sum_total=0
    sum_avg=0

    for run in $(seq 1 $N_RUNS); do
        line=$("$EXECUTABLE" "$TMP_CONF" 2>/dev/null \
               | grep '^\[timer\] main loop:')

        total=$(echo "$line" | grep -oP '(?<=total=)[0-9.]+')
        avg=$(echo   "$line" | grep -oP '(?<=avg=)[0-9.]+')

        sum_total=$(awk -v a="$sum_total" -v b="$total" 'BEGIN{printf "%.6f", a+b}')
        sum_avg=$(  awk -v a="$sum_avg"   -v b="$avg"   'BEGIN{printf "%.6f", a+b}')
    done

    mean_total=$(awk -v s="$sum_total" -v n="$N_RUNS" 'BEGIN{printf "%.4f", s/n}')
    mean_avg=$(  awk -v s="$sum_avg"   -v n="$N_RUNS" 'BEGIN{printf "%.6f", s/n}')

    printf "%-6d  %-14s  %-14s\n" "$nx" "$mean_total" "$mean_avg" | tee -a "$OUT_FILE"
done

echo "Results saved to $OUT_FILE"
