#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
repo_root=$(cd ../../.. && pwd)
export PYTHONPATH="$repo_root/kce/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MPLBACKEND=Agg

stage=0
stop_stage=1
librispeech_dir=
config=conf/train.yaml
exp_dir=exp/kce
gpus=0,1,2,3,4,5,6,7
seed=4011
nj=8
valid_samples=500
subsets="train-clean-360 train-other-500 train-clean-100"
dump=data/dump

. local/parse_options.sh

resources=../../librimix/dae-tse/data/text_cue

# A step is reusable only after its command has completed successfully.
prepare_step() {
  local output=$1
  local label=$2
  shift 2

  if [ -f "${output}.done" ] && [ -f "$output" ]; then
    echo "Reuse $label: $output"
    return
  fi

  rm -f -- "${output}.done"
  echo "Prepare $label: $output"
  "$@"
  touch "${output}.done"
}

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  [ -d "$librispeech_dir" ] || { echo "Set --librispeech_dir to LibriSpeech" >&2; exit 1; }

  for subset in $subsets; do
    prepare_step "$dump/wavscp/$subset.scp" "$subset WAV List" \
      python local/generate_wavscp.py --librispeech_dir "$librispeech_dir" \
      --subset "$subset" --output_dir "$dump/wavscp"

    prepare_step "$dump/duration/$subset.dur" "$subset Duration" \
      bash local/wav_to_duration.sh --nj "$nj" \
      "$dump/wavscp/$subset.scp" "$dump/duration/$subset.dur"

    prepare_step "$dump/text_label/$subset.jsonl" "$subset Text Labels" \
      python local/generate_text_label.py --librispeech_dir "$librispeech_dir" \
      --subset "$subset" --output_dir "$dump/text_label" \
      --p2idx_path "$resources/phoneme2int.txt" \
      --lexicon_path "$resources/word2lexicon.txt"

    prepare_step "$dump/datalist/$subset.jsonl" "$subset JSONL" \
      python local/generate_datalist.py --subset "$subset" \
      --wavscp_dir "$dump/wavscp" --text_label_dir "$dump/text_label" \
      --output_dir "$dump/datalist"
  done

  prepare_step "$dump/datalist/train-860.jsonl" "Combined Training JSONL" \
    python local/merge_datalist.py --datalist_dir "$dump/datalist" \
    --duration_dir "$dump/duration" --subsets train-clean-360 train-other-500 \
    --max_duration 20 --output_file "$dump/datalist/train-860.jsonl"

  prepare_step "$dump/datalist/valid.jsonl" "Validation JSONL" \
    python local/merge_datalist.py --datalist_dir "$dump/datalist" \
    --duration_dir "$dump/duration" --subsets train-clean-100 \
    --max_duration 20 --num_samples "$valid_samples" \
    --output_file "$dump/datalist/valid.jsonl"

  mkdir -p "$dump/speaker_label"
  prepare_step "$dump/speaker_label/spk2cls.json" "Speaker Labels" \
    python local/generate_speaker_label.py \
    --train_datalist "$dump/datalist/train-860.jsonl" \
    --valid_datalist "$dump/datalist/train-clean-100.jsonl" \
    --output_dir "$dump/speaker_label"
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  mkdir -p "$exp_dir"
  start_epoch=0
  if [ -d "$exp_dir" ]; then
    latest=$(find "$exp_dir" -maxdepth 1 -name 'epoch_[0-9]*.pt' -printf '%f\n' | sort -V | tail -n 1)
    if [ -n "$latest" ]; then
      epoch=${latest#epoch_}
      start_epoch=$((${epoch%.pt} + 1))
    fi
  fi

  IFS=',' read -r -a gpu_ids <<< "$gpus"
  CUDA_VISIBLE_DEVICES="$gpus" torchrun --standalone \
    --nproc_per_node="${#gpu_ids[@]}" -m kce.bin.train \
    --config "$config" --exp_dir "$exp_dir" \
    --start_epoch "$start_epoch" --seed "$seed"
fi
