#!/usr/bin/env python3
"""Demo helper: time-aligned KCE mixture-keyword cross-attention heatmaps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

# Special tokens used by KCE / batch inference
PSOK, PEOK = 71, 72


def load_phoneme_map(path: str) -> list[str]:
    idx2p = []
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            idx2p.append(parts[0])
    return idx2p


def trim_special_kw(att: np.ndarray, kw_ids: list[int]):
    """Trim leading/trailing psok/peok on the keyword axis. att: [T, K]."""
    lo, hi = 0, len(kw_ids)
    if hi > 0 and kw_ids[0] in (PSOK, PEOK):
        lo = 1
    if hi - lo > 0 and kw_ids[hi - 1] in (PSOK, PEOK):
        hi -= 1
    if hi <= lo:
        return att, kw_ids
    return att[:, lo:hi], kw_ids[lo:hi]


def plot_attention_heatmap(
    waveform: np.ndarray,
    sample_rate: int,
    attention: np.ndarray,
    frame_times_sec: np.ndarray,
    kw_ids: list[int],
    idx2p: list[str],
    out_path: str,
    title: str = "",
    dpi: int = 150,
):
    """Two-panel figure: waveform (+ max-attn curve) and phoneme×time heatmap.

    ``attention``: [T, K], ``frame_times_sec``: [T] RF-center times.
    """
    att, kw_ids = trim_special_kw(np.asarray(attention), list(kw_ids))
    times = np.asarray(frame_times_sec, dtype=np.float64)
    if att.shape[0] != times.shape[0]:
        raise ValueError(
            f"attention T={att.shape[0]} != frame_times {times.shape[0]}")

    labels = [idx2p[i] if i < len(idx2p) else str(i) for i in kw_ids]
    n_kw = max(len(labels), 1)
    dur = float(len(waveform)) / float(sample_rate)
    if times.size:
        half = 0.5 * (times[1] - times[0]) if times.size > 1 else 0.02
        t0, t1 = float(times[0] - half), float(times[-1] + half)
    else:
        t0, t1 = 0.0, dur

    wav_t = np.arange(len(waveform), dtype=np.float64) / sample_rate
    max_attn = att.max(axis=1) if att.size else np.zeros_like(times)
    if max_attn.size and max_attn.max() > 0:
        max_attn = max_attn / max_attn.max()

    # Grow figure with keyword length so y-tick labels are not cramped.
    # (~0.30in per phoneme row; clamp for very short / very long cues.)
    wav_h = 1.3
    hm_h = float(np.clip(n_kw * 0.30, 2.2, 28.0))
    fig_w = float(np.clip(10.0 + 0.12 * max(dur, 1.0), 10.0, 18.0))
    fig_h = wav_h + hm_h + 0.7
    y_fs = 8 if n_kw <= 40 else (7 if n_kw <= 80 else 6)

    fig, (ax_wav, ax_hm) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h), sharex=True,
        gridspec_kw={"height_ratios": [wav_h, hm_h], "hspace": 0.08},
    )
    ax_wav.plot(wav_t, waveform, color="#444444", lw=0.6, label="mixture")
    if times.size:
        ax_wav.plot(
            times, max_attn * (np.max(np.abs(waveform)) + 1e-8),
            color="#c44e52", lw=1.2, alpha=0.85, label="max attn (norm)",
        )
    ax_wav.set_ylabel("amp")
    ax_wav.set_xlim(0.0, max(dur, t1))
    ax_wav.legend(loc="upper right", fontsize=8, frameon=False)
    if title:
        ax_wav.set_title(title)

    # Heatmap: rows = phonemes (keyword axis flipped: first phone at bottom),
    # cols = time
    hm = att.T[::-1]
    labels = labels[::-1]
    im = ax_hm.imshow(
        hm,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="copper",
        extent=[t0, t1, len(labels), 0],
    )
    ax_hm.set_yticks(np.arange(len(labels)) + 0.5)
    ax_hm.set_yticklabels(labels, fontsize=y_fs)
    ax_hm.set_xlabel("time (s)")
    ax_hm.set_ylabel("keyword phoneme")
    fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.02, label="attn")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.no_grad()
def attn_frame_center_times_sec(n_attn: int, frame_shift_s: float = 0.01):
    """Map attention-frame index -> wall-clock seconds (RF center).

    Derived from fbank hop and KCE speech-frontend temporal strides.
    """
    centers = 4 * np.arange(n_attn, dtype=np.float64) + 3
    return centers * frame_shift_s


def attach_frame_times(attn_pack: dict) -> dict:
    """Add ``frame_times_sec`` to an encoder attention pack (demo-side)."""
    frame_shift_s = float(attn_pack.get("frame_shift_s", 0.01))
    times = []
    for att in attn_pack["attention"]:
        n_t = int(att.shape[0])
        times.append(torch.tensor(
            attn_frame_center_times_sec(n_t, frame_shift_s),
            dtype=torch.float32))
    out = dict(attn_pack)
    out["frame_times_sec"] = times
    return out


@torch.no_grad()
def extract_attention_from_encoder(asr_encoder, mix_wav: torch.Tensor,
                                   phn_label: torch.Tensor, phn_len: torch.Tensor,
                                   attn_layer: int = -1):
    """Thin wrapper: TCASREncoder(..., return_attention=True)."""
    _emb, _pred, pack = asr_encoder(
        mix_wav, [phn_label, phn_len],
        return_attention=True, attn_layer=attn_layer)
    return attach_frame_times(pack)


def _build_kw_tensor(converter, text: str, device: torch.device):
    from wesep.dataset.processor import flatten_list

    result = converter.convert(text)
    kw = [PSOK, result["phn_label"], PEOK]
    ids = flatten_list(kw)
    if len(ids) <= 2:
        raise ValueError("Keyword text produced an empty phoneme sequence.")
    phn_label = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    phn_len = torch.tensor([phn_label.shape[-1]], dtype=torch.long, device=device)
    return phn_label, phn_len, ids


def main():
    parser = argparse.ArgumentParser(
        description="Plot time-aligned KCE cross-attention heatmap (demo helper)")
    parser.add_argument("--exp_dir", default="exp/backbone",
                        help="Backbone exp (loads asr encoder via config)")
    parser.add_argument("--wav", default=None, help="Mixture wav path")
    parser.add_argument("--uid", default=None, help="LibriMix mixture uid")
    parser.add_argument("--wav_scp", default="data/clean/test/wav.scp")
    parser.add_argument("--keyword", required=True, help="Enroll keyword text")
    parser.add_argument(
        "--phoneme_map", default="data/text_cue/phoneme2int.txt")
    parser.add_argument(
        "--lexicon", default="data/text_cue/word2lexicon.txt")
    parser.add_argument("--out", default="demo/extracted/attention.png")
    parser.add_argument("--attn_layer", type=int, default=-1)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    recipe = os.path.realpath(os.path.join(demo_dir, ".."))
    os.chdir(recipe)
    sys.path.insert(0, os.path.join(recipe, "local"))
    sys.path.insert(0, os.path.realpath(os.path.join(recipe, "../../..")))

    from text2phoneme import Text2Phoneme, prepare_keyword_text
    from wesep.models import get_model
    from wesep.utils.checkpoint import load_pretrained_model
    from wesep.utils.utils import parse_config_or_kwargs

    wav_path = args.wav
    if args.uid:
        with open(args.wav_scp, encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split()
                if parts and parts[0] == args.uid:
                    wav_path = parts[1]
                    break
        if not wav_path:
            raise SystemExit(f"uid not found in {args.wav_scp}: {args.uid}")

    if not wav_path or not os.path.isfile(wav_path):
        raise SystemExit(f"Missing mixture wav: {wav_path}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    configs = parse_config_or_kwargs(os.path.join(args.exp_dir, "config.yaml"))
    if "spk_model_init" in configs["model_args"]["tse_model"]:
        configs["model_args"]["tse_model"]["spk_model_init"] = False
    model = get_model(configs["model"]["tse_model"])(
        **configs["model_args"]["tse_model"])
    ckpt = os.path.join(args.exp_dir, "models", "checkpoint_150.pt")
    load_pretrained_model(model, ckpt)
    model = model.to(device).eval()
    asr = model.asr_encoder
    if asr is None:
        raise SystemExit("Model has no asr_encoder (TCASREncoder)")

    converter = Text2Phoneme(args.phoneme_map, args.lexicon)
    try:
        keyword = prepare_keyword_text(args.keyword)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    phn_label, phn_len, ids = _build_kw_tensor(converter, keyword, device)

    wav, sr = torchaudio.load(wav_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000
    wav = wav.mean(dim=0, keepdim=True).to(device)

    pack = extract_attention_from_encoder(
        asr, wav, phn_label, phn_len, attn_layer=args.attn_layer)
    att = pack["attention"][0].numpy()
    times = pack["frame_times_sec"][0].numpy()
    dur = wav.shape[-1] / 16000.0
    print(json.dumps({
        "wav": wav_path,
        "duration_sec": round(dur, 4),
        "attn_frames": int(att.shape[0]),
        "t_first": round(float(times[0]), 4) if times.size else None,
        "t_last": round(float(times[-1]), 4) if times.size else None,
        "frame_shift_s": pack["frame_shift_s"],
        "layer_idx": pack["layer_idx"],
    }, indent=2))

    idx2p = load_phoneme_map(args.phoneme_map)
    pcm = wav[0].detach().cpu().numpy()
    title = f"keyword: {args.keyword}"
    if args.uid:
        title = f"{args.uid} | {title}"
    plot_attention_heatmap(
        pcm, 16000, att, times, ids, idx2p, args.out, title=title)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
