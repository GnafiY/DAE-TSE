# Interactive Gradio demo (LibriMix)

Small interactive toy for keyword-guided extraction.
Adapted from the WeSep two-speaker TSE demo:
https://huggingface.co/spaces/wenet-e2e/wesep-tse-2speaker-demo

**English only.** Other languages have not been supported up to now.

## Setup

From the repo root (after the main inference env is installed and `exp/` is linked):

```bash
pip install -r requirements.txt
cd recipes/librimix/dae-tse
# needs data/clean/test/wav.scp (same as infer.sh stage 1)
bash demo/run_demo.sh
```

Open the printed Gradio URL (default `http://0.0.0.0:7860`).

Optional:

```bash
SERVER_PORT=7861 bash demo/run_demo.sh
EXP_DIR=exp/backbone bash demo/run_demo.sh
WAV_SCP=data/clean/test/wav.scp \
TEXT_ENROLL=data/text_cue/testset/kw-4_seed-42.jsonl \
  bash demo/run_demo.sh
# offline g2p / nltk:
# export NLTK_DATA=/path/to/nltk_data
```

## How to use

### LibriMix clean (recommended)

1. Open the **LibriMix clean** tab.
2. Enter a mixture **uid** (or pick one from the example dropdown) and click
   **Load uid**.
3. The UI loads the mix wav and shows each speaker’s **full transcript** —
   copy a short consecutive English phrase into the keyword box.
4. Click **Extract** — extracted wav + time-aligned attention heatmap.

### OOD upload

**English only.** Upload a mixture and type an English keyword yourself.
Other languages are not recommended. Quality outside LibriMix is not guaranteed.

### GPU (tested on Tesla V100)

- After load: ≈0.6 GB allocated
- Peak with extract + heatmap on longest ~14 s mixes: ≈**3.0 GB** allocated

## Notes

- Uses the same backbone/KCE checkpoints as batch `infer.sh`
- Text → phoneme via `../local/text2phoneme.py` and `../data/text_cue/`
- Runtime dirs `audios/`, `extracted/`, `.gradio/` are local-only (gitignored)
