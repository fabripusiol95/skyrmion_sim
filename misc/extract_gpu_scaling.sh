#!/bin/bash
# extract_gpu_scaling.sh
#
# Reads all nsys-rep files in results/pin_scaling/, extracts GPU time (%)
# for force_ss_k, force_pin_k, and force_pin_cll_k, and writes a CSV.
#
# Usage: ./extract_gpu_scaling.sh

set -euo pipefail

RESULTS="$(cd "$(dirname "$0")" && pwd)/results/pin_scaling"
CSV="$RESULTS/gpu_pct.csv"

[[ -d "$RESULTS" ]] || { echo "ERROR: $RESULTS not found"; exit 1; }

echo "N,mode,kernel,gpu_pct,avg_us,total_us,instances" > "$CSV"

# Process files in ascending N order
for rep in $(ls "$RESULTS"/*.nsys-rep | sort -t'N' -k2 -V); do
    filename="$(basename "$rep" .nsys-rep)"  # e.g. N1024_naive
    N="${filename%%_*}";   N="${N#N}"        # strip leading N -> 1024
    mode="${filename##*_}"                    # naive or cll

    nsys stats --report gpukernsum "$rep" 2>/dev/null \
        | grep -E "force_ss_k|force_pin_k|force_pin_cll_k" \
        | tr -d ',' \
        | awk -v N="$N" -v mode="$mode" -v csv="$CSV" '
          {
              pct=$1; total_ns=$2; instances=$3; avg_ns=$4
              # Kernel name starts at field 15 (after 8 stats + 3 GridXYZ + 3 BlockXYZ)
              name=$15
              if      (name ~ /^force_pin_cll_k/) kernel="force_pin_cll_k"
              else if (name ~ /^force_pin_k/)     kernel="force_pin_k"
              else if (name ~ /^force_ss_k/)      kernel="force_ss_k"
              else                               kernel=name
              printf "%s,%s,%s,%.1f,%.3f,%.1f,%d\n",
                     N, mode, kernel,
                     pct, avg_ns/1000.0, total_ns/1000.0, instances >> csv
          }'
done

echo "Written: $CSV"
echo ""
column -t -s ',' "$CSV"
