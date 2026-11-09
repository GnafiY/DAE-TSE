#!/usr/bin/env bash
# Source: https://github.com/GnafiY/wekws/blob/38e88b74e52e5fa225a9641f32722d0bf4b2637b/tools/wav_to_duration.sh
# Adapted for this recipe; see repository LICENSE.
# Adapted from WeKws tools/wav_to_duration.sh.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nj=8

. "$script_dir/parse_options.sh"

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 [--nj N] wav.scp output.dur [log_dir]" >&2
  exit 1
fi

input=$1
output=$2
log_dir=${3:-${output%.dur}/tmp}
if ! [[ "$nj" =~ ^[1-9][0-9]*$ ]]; then
  echo "--nj must be a positive integer" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$output")" "$log_dir"
rm -f -- "$log_dir"/wav_*.slice "$log_dir"/wav_*.shape
split --additional-suffix=.slice -d -n "l/$nj" -- "$input" "$log_dir/wav_"

echo "Compute duration: $input ($nj processes)"
pids=()
for slice in "$log_dir"/wav_*.slice; do
  name=$(basename "$slice" .slice)
  python "$script_dir/wav2dur.py" "$slice" "$log_dir/$name.shape" \
    > "$log_dir/$name.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
  echo "Duration worker failed; see $log_dir/wav_*.log" >&2
  exit 1
fi

cat "$log_dir"/wav_*.shape > "$log_dir/merged.dur"
mv -- "$log_dir/merged.dur" "$output"
count=$(wc -l < "$output")
printf 'Duration saved: %s (%s utterances)\n' "$output" "$count"
