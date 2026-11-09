#!/bin/bash
# Launch the local DAE-TSE Gradio demo (LibriMix-oriented).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RECIPE_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${RECIPE_DIR}"

. ./path.sh || exit 1

EXP_DIR=${EXP_DIR:-exp/backbone}
SERVER_NAME=${SERVER_NAME:-0.0.0.0}
SERVER_PORT=${SERVER_PORT:-7860}

if [ ! -f "${EXP_DIR}/config.yaml" ] || [ ! -f "${EXP_DIR}/models/checkpoint_150.pt" ]; then
  echo "ERROR: missing pretrained backbone under ${EXP_DIR}"
  echo "  ln -s /path/to/Pretrained/backbone exp/backbone"
  echo "  ln -s /path/to/Pretrained/kce      exp/kce"
  exit 1
fi

if [ ! -f exp/kce/epoch_149.pt ]; then
  echo "ERROR: missing KCE checkpoint (exp/kce/epoch_149.pt)"
  exit 1
fi

# Optional: export NLTK_DATA=/path/to/nltk_data if g2p_en needs it offline
export DAE_TSE_EXP_DIR=$(realpath "${EXP_DIR}")

WAV_SCP=${WAV_SCP:-data/clean/test/wav.scp}
TEXT_ENROLL=${TEXT_ENROLL:-data/text_cue/testset/kw-4_seed-42.jsonl}

echo "Starting demo with exp=${DAE_TSE_EXP_DIR} on ${SERVER_NAME}:${SERVER_PORT}"
echo "  wav_scp=${WAV_SCP}  text_enroll=${TEXT_ENROLL}"
python demo/app.py \
  --exp_dir "${DAE_TSE_EXP_DIR}" \
  --wav_scp "${WAV_SCP}" \
  --text_enroll "${TEXT_ENROLL}" \
  --server_name "${SERVER_NAME}" \
  --server_port "${SERVER_PORT}" \
  "$@"
