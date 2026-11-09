# DAE-TSE: Detect, Attend and Extract — Keyword Guided Target Speaker Extraction

<p align="center">
  <a href="https://ijcai.org/proceedings/2026/0644.pdf"><img src="https://img.shields.io/badge/IJCAI%202026-Paper-00599C.svg" alt="IJCAI 2026 Paper"></a>
  <a href="https://huggingface.co/GnafiY/DAE-TSE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Pretrained-yellow" alt="HuggingFace"></a>
  <a href="https://gnafiy.github.io/DAE-TSE_demo"><img src="https://img.shields.io/badge/Demo-Page-blue" alt="Demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

## Table of Contents

- [News](#news)
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Pretrained Models](#pretrained-models)
- [Data](#data)
- [Interactive Demo](#interactive-demo-interactive)
- [Training](#training)
- [Inference](#inference)
- [Citation](#citation)
- [License](#license)

## News

Dates below are in **China Standard Time (UTC+8)**.

- **[2026-11-09]** 📦 Official KCE and backbone checkpoints and their whole-utterance training recipes are available.
- **[2026-11-02]** 🚀 Model training code has been integrated into [WeSep](https://github.com/wenet-e2e/wesep); KCE-related code is being organized and will be released soon.
- **[2026-08-13]** 🎮 Interactive Gradio demo released (LibriMix uid lookup + OOD upload).
- **[2026-08-06]** 📦 Official inference code and pretrained weights released.

## Overview

**DAE-TSE** extracts a target speaker from multi-talker mixtures using only **keywords** (partial transcription) spoken by that speaker — no clean enrollment utterance is required.

It follows the **Detect-Attend-Extract** paradigm:

1. **Detect** whether the given keywords appear in the mixture
2. **Attend** to the corresponding speaker via mixture-keyword cross-attention
3. **Extract** the full target speech with a text-aware TSE backbone

The system has two components: a **Keyword-guided Cue Encoder (KCE)** that produces a speaker cue from the mixture and keywords, and a **TSE backbone** (text-aware BSRNN) that performs extraction. On LibriMix with 4 keywords (~28% of the full transcript), the released checkpoint reaches **SI-SNRi = 16.45 dB**.

<p align="center">
  <img src="resources/overview.jpg" alt="DAE-TSE Overview" width="80%">
</p>

This repository provides the paper inference code and migrated whole-utterance training recipes for KCE and the TSE backbone. Built upon [WeSep](https://github.com/wenet-e2e/wesep).

## Environment Setup

Requirements: **Python 3.10+**, **CUDA-capable GPU**, and **ffmpeg**.

```bash
git clone https://github.com/GnafiY/DAE-TSE
cd DAE-TSE

conda create -n dae-tse python=3.10 -y
conda activate dae-tse
conda install -c conda-forge ffmpeg -y

# 1) Install the CUDA 12.6 PyTorch stack (exact versions)
pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.9.1+cu126 torchaudio==2.9.1+cu126 torchcodec==0.9.1+cu126

# 2) Install inference, training and demo dependencies + this repo
pip install -r requirements.txt
pip install -e .
pip install -e kce
```

> Tip: Install `torch` / `torchaudio` / `torchcodec` from the [official PyTorch CUDA 12.6 index](https://download.pytorch.org/whl/cu126) first (step 1). This release expects **`torch==2.9.1+cu126`**, **`torchaudio==2.9.1+cu126`**, and **`torchcodec==0.9.1+cu126`**. Then run `pip install -r requirements.txt`.

Verify:

```bash
python -c "import torch, torchaudio, torchcodec, wesep, kce; print(torch.__version__, torchaudio.__version__, torch.cuda.is_available())"
```

## Pretrained Models

Download from [HuggingFace](https://huggingface.co/GnafiY/DAE-TSE) and place (or soft-link) under the recipe:

```bash
cd recipes/librimix/dae-tse
mkdir -p exp
ln -s /path/to/Pretrained/backbone exp/backbone
ln -s /path/to/Pretrained/kce      exp/kce
```

Expected layout:

```text
recipes/librimix/dae-tse/exp/
├── backbone/
│   ├── config.yaml
│   └── models/checkpoint_150.pt
└── kce/
    ├── epoch_149.pt
    ├── model.yaml
    └── data.yaml
```

`config.yaml` should point `model_args.tse_model.asr_encoder.checkpoint_path` to the KCE checkpoint (e.g. `exp/kce/epoch_149.pt`). Keep `model.yaml` / `data.yaml` next to that checkpoint.

## Data

### Keyword cues (shipped)

This repo ships **test-set keyword cues**:

```text
recipes/librimix/dae-tse/data/text_cue/testset/
├── kw-1_seed-42.jsonl
├── ...
└── kw-4_seed-42.jsonl
```

### LibriMix audio (prepare via WeSep-style scripts)

Download / build [Libri2Mix](https://github.com/JorisCos/LibriMix) (e.g. `wav16k/min`), then run **stage 1** of `infer.sh` to generate `data/clean/test/wav.scp`:

```bash
cd recipes/librimix/dae-tse
bash infer.sh --stage 1 --stop-stage 1 \
    --Libri2Mix_dir /path/to/Libri2Mix
```

Each `wav.scp` line is `key mix.wav s1.wav s2.wav`. DAE-TSE does **not** need speaker enrollment maps — the cue is the keyword JSONL.

This calls `local/prepare_data.sh` (WeSep-style meta preparation).

### Text → phoneme helper

Keyword cues are phoneme-id sequences. Convert English text with:

```bash
cd recipes/librimix/dae-tse
pip install g2p_en   # if not already installed
python local/text2phoneme.py "Hey Siri open the door" --show_phones
```

Resources used (shipped under `data/text_cue/`):

- `phoneme2int.txt` — phoneme → id (must match KCE)
- `word2lexicon.txt` — optional word cache for speed

## Interactive Demo (Interactive)

An interactive Gradio demo for keyword-guided extraction on **LibriMix mix-clean**
(enter a mixture uid → show full s1/s2 transcripts to copy keywords).
OOD uploads are not guaranteed.

**Prerequisite:** prepare LibriMix inference data first (same as batch inference
stage 1). The LibriMix tab needs `recipes/librimix/dae-tse/data/clean/test/wav.scp`
— see [LibriMix audio](#librimix-audio-prepare-via-wesep-style-scripts) above.
Without that list, uid lookup will not work (OOD upload tab can still run).

```bash
# main env already set up; checkpoints linked under recipes/librimix/dae-tse/exp/
# and LibriMix wav.scp prepared via infer.sh --stage 1
cd recipes/librimix/dae-tse
bash demo/run_demo.sh
```

See [recipes/librimix/dae-tse/demo/README.md](recipes/librimix/dae-tse/demo/README.md).

## Training

The recipes migrate the original KCE and WeSep training code into the official
inference repository. The published `AEDKWSASRPhone`, `NetModules`, DAE-BSRNN
architecture, parameter names and original phone IDs are retained. These recipes
support whole utterances; chunk training is outside this release.

Use the environment and single `requirements.txt` described above. Make sure it
imports this repository's `kce` package rather than a separate research KCE
implementation. G2p-en may download NLTK resources on first use; offline
machines can provide these resources through `NLTK_DATA`.

### Stage A: Train KCE On LibriSpeech

```bash
cd recipes/librispeech/kce

# LibriSpeech root contains train-clean-360, train-other-500, train-clean-100.
bash run.sh --stage 0 --stop_stage 0 \
  --librispeech_dir /path/to/LibriSpeech

bash run.sh --stage 1 --stop_stage 1 \
  --config conf/train.yaml --exp_dir exp/kce \
  --gpus 0,1,2,3,4,5,6,7
```

Rerun stage 1 with the same experiment directory to continue from the latest
completed epoch, including the original optimizer and update count. KCE epochs
are zero-based: a 150-epoch run produces `epoch_149.pt`. The directory also
contains `model.yaml`, `data.yaml`, `exp.yaml`, per-rank training logs, loss
curves and optimizer checkpoints. Keep the model/data YAML files beside a KCE
checkpoint when passing it to the backbone.

Data preparation retains the original KCE format: one plain JSONL file per
LibriSpeech subset, with one utterance per line. The default layout is:

```text
data/dump/datalist/
├── train-clean-100.jsonl
├── train-clean-360.jsonl
├── train-other-500.jsonl
├── train-860.jsonl        # Combined train-clean-360 + train-other-500, <=20 s.
└── valid.jsonl            # 500 train-clean-100 records, <=20 s.
```

Each record contains `key` (utterance ID), `sph` (audio path), `phn_label`
(original phone IDs grouped by word), `kw_candidate` (candidate word indices),
and `text`. Only phoneme labels and keyword candidates are generated.
Speaker-class IDs are stored separately in `data/dump/speaker_label/spk2cls.json`.
Mixing and keyword sampling happen during training; no pre-rendered mixture
bundle is required.

### Stage B: Train The Whole-Utterance TSE Backbone

The published KCE may be downloaded for this stage, or replaced by your own
completed KCE run with its adjacent model/data YAML files using `--kce_dir /path/to/exp/kce`.

```bash
cd recipes/librimix/dae-tse

# Official Libri2Mix wav16k/min, with train-100/dev/test and clean references.
bash run.sh --stage 0 --stop_stage 0 \
  --Libri2Mix_dir /path/to/Libri2Mix

bash run.sh --stage 1 --stop_stage 1 \
  --config confs/dae-bsrnn.yaml --exp_dir exp/dae-bsrnn \
  --gpus 0,1,2,3,4,5,6,7

# Evaluate the final epoch with the existing four-keyword paper protocol.
bash run.sh --stage 2 --stop_stage 2 --exp_dir exp/dae-bsrnn
```

Stage 0 downloads the already published NeMo transcripts and KCE resources.
It regenerates the original phoneme cue lists using the supplied word cache.
Keywords come from NeMo transcriptions of the cropped Libri2Mix reference
sources, as in the original pipeline; no new ASR run is necessary. To work offline,
provide these resources first and pass `--download false --transcript /path/to/transcript_nemo.jsonl`.

## Inference

```bash
conda activate dae-tse
cd recipes/librimix/dae-tse

# stage 1: prepare LibriMix lists  |  stage 2: extract + SI-SNRi
bash infer.sh --stage 1 --stop-stage 2 \
    --Libri2Mix_dir /path/to/Libri2Mix \
    --nj 1 --infer_tag kw4 \
    --exp_dir exp/backbone \
    --test_text_enroll data/text_cue/testset/kw-4_seed-42.jsonl
```

If data is already prepared, run inference only:

```bash
bash infer.sh --stage 2 --stop-stage 2 \
    --nj 1 --infer_tag kw4 \
    --exp_dir exp/backbone \
    --test_text_enroll data/text_cue/testset/kw-4_seed-42.jsonl
```

Stage 2 reports SI-SNR / SI-SNRi. With the released checkpoint and `kw-4_seed-42.jsonl`, you should obtain **Overall SI-SNRi ≈ 16.45**.

Outputs:

```text
${exp_dir}/inference/${infer_tag}/infer.jsonl   # per-utterance scores
${exp_dir}/inference/${infer_tag}/audio/        # optional wavs (--save_results true)
```

Use `--gpus 2,3` for evaluation on the selected GPUs. The script automatically
creates test shards under `${exp_dir}/inference/${infer_tag}/splitN/` and merges
scores into `infer.jsonl`. For multiple GPUs, each rank's audio and logs stay in
`splitN/<rank>/`; single-GPU audio stays in `audio/`. The legacy `--nj N` option
selects visible GPUs `0..N-1` when `--gpus` is omitted. `run.sh` stage 2 also forwards
its `--gpus` selection. Set `--save_results false` to skip writing waveforms.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{ijcai2026p644,
  title     = {Detect, Attend and Extract: Keyword Guided Target Speaker Extraction},
  author    = {Li, Haoyu and Xi, Yu and Jiang, Yidi and Wang, Shuai and Knill, Kate and Gales, Mark and Li, Haizhou and Yu, Kai},
  booktitle = {Proceedings of the Thirty-Fifth International Joint Conference on
               Artificial Intelligence, {IJCAI-26}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Diego Calvanese},
  pages     = {5784--5792},
  year      = {2026},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2026/644},
  url       = {https://doi.org/10.24963/ijcai.2026/644},
}

```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
