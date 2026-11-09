#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
repo_root=$(cd ../../.. && pwd)
export PYTHONPATH="$repo_root/kce/src:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MPLBACKEND=Agg

stage=0
stop_stage=1
Libri2Mix_dir=
config=confs/dae-bsrnn.yaml
exp_dir=exp/dae-bsrnn
gpus=0,1,2,3,4,5,6,7
checkpoint=
kce_dir=exp/kce
transcript=data/text_cue/transcript_nemo.jsonl
download=true

. local/parse_options.sh

if [ "$stage" -le 0 ] && [ "$stop_stage" -ge 0 ]; then
  python local/prepare_training_data.py \
    --mix_data_path "$Libri2Mix_dir/wav16k/min"

  if $download; then
    python local/download_pretrained.py --output "$kce_dir" --transcript "$transcript"
  fi

  mkdir -p data/text_cue/dump/ready
  python local/postprocess_asr_results.py --asr_transcript "$transcript" \
    --output_dir data/text_cue/dump/ready
fi

if [ "$stage" -le 1 ] && [ "$stop_stage" -ge 1 ]; then
  if [ -z "$checkpoint" ] && [ -f "$exp_dir/models/latest_checkpoint.pt" ]; then
    checkpoint="$exp_dir/models/latest_checkpoint.pt"
  fi

  IFS=',' read -r -a gpu_ids <<< "$gpus"
  logical_gpus=$(seq -s, 0 "$((${#gpu_ids[@]} - 1))")
  resume_args=()
  [ -z "$checkpoint" ] || resume_args=(--checkpoint "$checkpoint")
  CUDA_VISIBLE_DEVICES="$gpus" torchrun --standalone \
    --nproc_per_node="${#gpu_ids[@]}" "$repo_root/wesep/bin/train.py" \
    --config "$config" --exp_dir "$exp_dir" --gpus "[$logical_gpus]" \
    --kce_checkpoint "$kce_dir/epoch_149.pt" \
    "${resume_args[@]}"
fi

if [ "$stage" -le 2 ] && [ "$stop_stage" -ge 2 ]; then
  bash infer.sh --stage 2 --stop-stage 2 --exp_dir "$exp_dir" --gpus "$gpus" \
    --checkpoint checkpoint_150.pt --infer_tag kw4 \
    --test_text_enroll data/text_cue/testset/kw-4_seed-42.jsonl
fi
