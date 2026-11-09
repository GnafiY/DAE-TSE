#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
repo_root=$(cd ../../.. && pwd)
export PYTHONPATH="$repo_root/kce/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=UTF-8

log() {
  echo "$(date '+%Y-%m-%dT%H:%M:%S') $*"
}
SECONDS=0

stage=1
stop_stage=2
nj=1
gpus=
data=data
fs=16k
min_max=min
noise_type=clean
Libri2Mix_dir=/path/to/Libri2Mix
exp_dir=
checkpoint=checkpoint_150.pt
save_results=true
infer_tag=
test_text_enroll=data/text_cue/testset/kw-4_seed-42.jsonl

. local/parse_options.sh

if [ -n "$gpus" ]; then
  IFS=',' read -r -a gpu_ids <<< "$gpus"
  nj=${#gpu_ids[@]}
else
  if ! [[ "$nj" =~ ^[1-9][0-9]*$ ]]; then
    echo "--nj must be a positive integer" >&2
    exit 1
  fi
  mapfile -t gpu_ids < <(seq 0 "$((nj - 1))")
fi

mix_data_path="$Libri2Mix_dir/wav${fs}/$min_max"
if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  log "Prepare LibriMix datasets ..."
  bash local/prepare_data.sh --mix_data_path "$mix_data_path" \
    --data "$data" --noise_type "$noise_type" --stage 1 --stop-stage 1
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  test_data="$data/$noise_type/test/wav.scp"
  config="$exp_dir/config.yaml"
  ckpt="$exp_dir/models/$checkpoint"
  infer_tag=${infer_tag:-$(date '+%Y-%m-%d_%H-%M-%S')}
  output_dir="$exp_dir/inference/$infer_tag"
  split_dir="$output_dir/split$nj"
  mkdir -p "$split_dir"

  # Generate fresh round-robin shards; callers do not need to prepare splitN.
  for ((rank=0; rank<nj; rank++)); do
    mkdir -p "$split_dir/$rank"
    : > "$split_dir/$rank/wav.scp"
  done
  awk -v nj="$nj" -v directory="$split_dir" \
    '{print > (directory "/" ((NR - 1) % nj) "/wav.scp")}' "$test_data"

  pids=()
  result_files=()
  cleanup() {
    local pid
    for pid in "${pids[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  log "Checkpoint: $ckpt; GPUs: ${gpu_ids[*]}"
  for ((rank=0; rank<nj; rank++)); do
    [ -s "$split_dir/$rank/wav.scp" ] || continue
    gpu=${gpu_ids[$rank]}
    rank_tag=$infer_tag
    if [ "$nj" -gt 1 ]; then
      rank_tag="$infer_tag/split$nj/$rank"
    fi
    result_files+=("$exp_dir/inference/$rank_tag/jsonl_split/infer.$gpu.jsonl")
    python "$repo_root/wesep/bin/infer.py" --config "$config" \
      --fs "$fs" --gpus "$gpu" --exp_dir "$exp_dir" --data_type raw \
      --test_data "$split_dir/$rank/wav.scp" --save_wav "$save_results" \
      --checkpoint "$ckpt" --infer_tag "$rank_tag" \
      --test_text_enroll "$test_text_enroll" &
    pids+=("$!")
  done

  if [ "${#pids[@]}" -eq 0 ]; then
    echo "No test mixtures in $test_data" >&2
    exit 1
  fi
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      unset 'pids[index]'
    else
      unset 'pids[index]'
      echo "Inference worker failed; stopping remaining workers" >&2
      exit 1
    fi
  done
  trap - EXIT INT TERM

  cat "${result_files[@]}" > "$output_dir/infer.jsonl"
  log "Merged results: $output_dir/infer.jsonl"
  python local/merge_jsonl.py --jsonl_or_dir "$output_dir/infer.jsonl"
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
