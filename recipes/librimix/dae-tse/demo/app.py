#!/usr/bin/env python3
"""Local Gradio demo for DAE-TSE (LibriMix uid lookup + OOD upload)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

import gradio as gr
import soundfile
import torch
import torchaudio

# Recipe root: recipes/librimix/dae-tse
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_RECIPE_DIR = os.path.realpath(os.path.join(_DEMO_DIR, ".."))
_REPO_ROOT = os.path.realpath(os.path.join(_RECIPE_DIR, "../../.."))

sys.path.insert(0, os.path.join(_RECIPE_DIR, "local"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "kce", "src"))

from text2phoneme import Text2Phoneme, prepare_keyword_text  # noqa: E402

from wesep.dataset.processor import flatten_list  # noqa: E402
from wesep.models import get_model  # noqa: E402
from wesep.utils.checkpoint import load_pretrained_model  # noqa: E402
from wesep.utils.utils import parse_config_or_kwargs  # noqa: E402

# Special tokens used by KCE / batch inference (psok / peok)
PSOK, PEOK = 71, 72

INPUT_DIR = os.path.join(_DEMO_DIR, "audios")
OUTPUT_DIR = os.path.join(_DEMO_DIR, "extracted")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_UID = "3575-170457-0028_7127-75947-0010"
DEFAULT_WAV_SCP = os.path.join(_RECIPE_DIR, "data", "clean", "test", "wav.scp")
DEFAULT_JSONL = os.path.join(
    _RECIPE_DIR, "data", "text_cue", "testset", "kw-4_seed-42.jsonl")


@dataclass
class MixtureEntry:
    mix_path: str
    s1_path: str = ""
    s2_path: str = ""
    s1_text: str = ""
    s2_text: str = ""
    s1_keywords: str = ""
    s2_keywords: str = ""


@dataclass
class LibriMixIndex:
    """uid -> mix path + speaker transcripts (from wav.scp + keyword jsonl)."""

    entries: dict[str, MixtureEntry] = field(default_factory=dict)
    wav_scp: str = ""
    jsonl: str = ""

    @classmethod
    def load(cls, wav_scp: str, jsonl: str) -> "LibriMixIndex":
        idx = cls(wav_scp=wav_scp, jsonl=jsonl)
        if not os.path.isfile(wav_scp):
            print(f"[demo] LibriMix wav.scp not found: {wav_scp}")
            return idx
        with open(wav_scp, encoding="utf-8") as fin:
            for line in fin:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                uid, mix = parts[0], parts[1]
                s1 = parts[2] if len(parts) > 2 else ""
                s2 = parts[3] if len(parts) > 3 else ""
                idx.entries[uid] = MixtureEntry(
                    mix_path=mix, s1_path=s1, s2_path=s2)

        if os.path.isfile(jsonl):
            with open(jsonl, encoding="utf-8") as fin:
                for line in fin:
                    obj = json.loads(line)
                    key = obj.get("key", "")
                    if "/" not in key:
                        continue
                    side, uid = key.split("/", 1)
                    if uid not in idx.entries:
                        continue
                    text = obj.get("normalized_text", "") or ""
                    kw = obj.get("keywords_text", "") or ""
                    if side == "s1":
                        idx.entries[uid].s1_text = text
                        idx.entries[uid].s1_keywords = kw
                    elif side == "s2":
                        idx.entries[uid].s2_text = text
                        idx.entries[uid].s2_keywords = kw
        else:
            print(f"[demo] keyword jsonl not found: {jsonl}")

        print(f"[demo] LibriMix index: {len(idx.entries)} mixtures "
              f"(scp={wav_scp}, jsonl={jsonl})")
        return idx

    def get(self, uid: str) -> MixtureEntry:
        uid = (uid or "").strip()
        if not uid:
            raise ValueError("Mixture uid is empty.")
        if uid not in self.entries:
            raise KeyError(
                f"Unknown mixture uid: {uid}. "
                f"Expected a key from {self.wav_scp or 'wav.scp'} "
                f"({len(self.entries)} entries loaded).")
        entry = self.entries[uid]
        if not os.path.isfile(entry.mix_path):
            raise FileNotFoundError(
                f"Mixture wav missing for {uid}: {entry.mix_path}")
        return entry


def resolve_exp_dir() -> str:
    exp = os.environ.get("DAE_TSE_EXP_DIR")
    if exp:
        return os.path.realpath(exp)
    return os.path.join(_RECIPE_DIR, "exp", "backbone")


def load_model(exp_dir: str):
    config = os.path.join(exp_dir, "config.yaml")
    ckpt = os.path.join(exp_dir, "models", "checkpoint_150.pt")
    if not os.path.isfile(config):
        raise FileNotFoundError(
            f"Missing {config}. Soft-link Pretrained backbone under "
            f"{os.path.join(_RECIPE_DIR, 'exp')} first.")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    configs = parse_config_or_kwargs(config)
    if "spk_model_init" in configs["model_args"]["tse_model"]:
        configs["model_args"]["tse_model"]["spk_model_init"] = False
    model = get_model(configs["model"]["tse_model"])(
        **configs["model_args"]["tse_model"])
    load_pretrained_model(model, ckpt)
    return model


def build_text2phoneme() -> Text2Phoneme:
    cue_dir = os.path.join(_RECIPE_DIR, "data", "text_cue")
    return Text2Phoneme(
        p2idx_path=os.path.join(cue_dir, "phoneme2int.txt"),
        lexicon_path=os.path.join(cue_dir, "word2lexicon.txt"),
    )


def text_to_kw_tensor(converter: Text2Phoneme, text: str, device: torch.device):
    # Punctuation stripped; non-English letters rejected upstream via
    # prepare_keyword_text (demo) or convert()'s normalize_text.
    result = converter.convert(text)
    # Match batch inference: [psok, word_phones..., peok] then flatten
    kw = [PSOK, result["phn_label"], PEOK]
    ids = flatten_list(kw)
    if len(ids) <= 2:
        raise ValueError("Keyword text produced an empty phoneme sequence.")
    phn_label = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    phn_len = torch.tensor([phn_label.shape[-1]], dtype=torch.long, device=device)
    return phn_label, phn_len, ids


def save_upload_as_16k_mono(audio, filename: str = "mixture.wav") -> str:
    """Accept Gradio Audio as filepath or (sample_rate, np.ndarray)."""
    out_path = os.path.join(INPUT_DIR, filename)
    if isinstance(audio, str):
        pcm, sample_rate = torchaudio.load(audio)
    else:
        sr, wav = audio
        soundfile.write(out_path, wav, sr)
        pcm, sample_rate = torchaudio.load(out_path)
    pcm = pcm.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        pcm = torchaudio.transforms.Resample(sample_rate, 16000)(pcm)
    torchaudio.save(out_path, pcm, 16000)
    return out_path


@torch.no_grad()
def extract_speech(model, converter, device, mixture_path: str, text: str,
                   heatmap_path: str | None = None):
    from heatmap import attach_frame_times, load_phoneme_map, plot_attention_heatmap

    audio, _ = torchaudio.load(mixture_path)
    audio = audio.to(device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    phn_label, phn_len, ids = text_to_kw_tensor(converter, text, device)

    t0 = time.perf_counter()
    want_attn = heatmap_path is not None
    out = model(
        audio,
        [],
        additional_input={
            "data_type": "TC-ASR",
            "return_attention": want_attn,
            "data": {
                "kw_label": phn_label,
                "kw_len": phn_len,
            },
        },
    )
    if want_attn:
        output, _, attn_pack = out
        attn_pack = attach_frame_times(attn_pack)
    else:
        output, _ = out
        attn_pack = None
    print(f"Extraction time: {time.perf_counter() - t0:.2f}s")

    heat_out = None
    if want_attn and attn_pack is not None:
        idx2p = load_phoneme_map(
            os.path.join(_RECIPE_DIR, "data", "text_cue", "phoneme2int.txt"))
        pcm = audio[0].detach().cpu().numpy()
        plot_attention_heatmap(
            pcm, 16000,
            attn_pack["attention"][0].numpy(),
            attn_pack["frame_times_sec"][0].numpy(),
            ids, idx2p, heatmap_path,
            title=f"keyword: {text.strip()}",
        )
        heat_out = heatmap_path
        print(f"Heatmap: {heatmap_path} "
              f"(T={attn_pack['attention'][0].shape[0]}, "
              f"t_last={float(attn_pack['frame_times_sec'][0][-1]):.3f}s)")

    return output.detach().cpu(), heat_out


def build_app(model, converter, device, index: LibriMixIndex):
    def lookup_uid(uid: str):
        try:
            entry = index.get(uid)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            raise gr.Error(str(exc)) from exc
        # Gradio only serves files under CWD / tmp / allowed_paths.
        # Copy mix into the recipe demo dir so Audio can display it.
        uid_safe = (uid or "").strip().replace("/", "_")
        local_mix = os.path.join(INPUT_DIR, f"{uid_safe}.wav")
        if (not os.path.isfile(local_mix) or
                os.path.getmtime(local_mix) < os.path.getmtime(entry.mix_path)):
            shutil.copy2(entry.mix_path, local_mix)
        hint = (
            f"Official keywords (from jsonl): "
            f"s1=`{entry.s1_keywords}` | s2=`{entry.s2_keywords}`"
        )
        return (
            local_mix,
            entry.s1_text,
            entry.s2_text,
            hint,
            entry.s1_keywords or "",
        )

    def run_extract(text_enroll: str, mixture):
        try:
            keyword = prepare_keyword_text(text_enroll)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        if mixture is None:
            raise gr.Error("Please provide a mixture audio.")
        mix_path = save_upload_as_16k_mono(mixture, "mixture.wav")
        heat_path = os.path.join(OUTPUT_DIR, "attention.png")
        try:
            speech, heat = extract_speech(
                model, converter, device, mix_path, keyword,
                heatmap_path=heat_path)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        out_path = os.path.join(OUTPUT_DIR, "speech.wav")
        soundfile.write(out_path, speech[0].numpy(), 16000)
        return out_path, heat

    n_mix = len(index.entries)
    scp_note = index.wav_scp or DEFAULT_WAV_SCP

    with gr.Blocks(
            title="DAE-TSE — Keyword Guided Target Speaker Extraction",
    ) as demo:
        gr.Markdown(
            f"""
# DAE-TSE — Keyword Guided Target Speaker Extraction (Interactive Demo)

Adapted from the
[WeSep two-speaker TSE demo](https://huggingface.co/spaces/wenet-e2e/wesep-tse-2speaker-demo).

- **Language:** English keywords / transcripts only. Other languages are currently not supported.
- **Libri2Mix mix-clean:** enter a mixture **uid**, load mix + full s1/s2
  transcripts, copy a short phrase into the keyword box, then extract.
  Requires `data/clean/test/wav.scp` (infer.sh stage 1). Currently indexed
  `{n_mix}` utts from `{scp_note}`.
- **OOD upload:** English mixture + English keyword typed by you.
  Quality outside Libri2Mix is **not guaranteed**.
- Extract also draws a **time-aligned** mixture-keyword cross-attention
  heatmap (x-axis in seconds).
- **GPU (Tesla V100):** model load ≈0.6 GB allocated; peak ≈**3 GB**
  on the longest Libri2Mix min test mixes (~14 s) with extract + heatmap.
  Shorter clips use less.
"""
        )

        with gr.Tabs():
            with gr.Tab("LibriMix clean"):
                if n_mix == 0:
                    gr.Markdown(
                        f"""
> **LibriMix data not ready.** Indexed **0** mixtures from `{scp_note}`.
> Prepare inference lists first (same as batch `infer.sh` stage 1), e.g.
> `bash infer.sh --stage 1 --stop-stage 1 --Libri2Mix_dir /path/to/Libri2Mix`,
> then restart this demo. Until then, use the **OOD upload** tab.
"""
                    )
                else:
                    gr.Markdown(
                        f"_Indexed **{n_mix}** mixtures from `{scp_note}`._"
                    )

                with gr.Row(elem_id="uid-load-row", equal_height=False):
                    uid_box = gr.Textbox(
                        value=DEFAULT_UID if DEFAULT_UID in index.entries else "",
                        label="Mixture uid",
                        placeholder="e.g. 3575-170457-0028_7127-75947-0010",
                        scale=2,
                        min_width=120,
                        container=False,
                    )
                    load_btn = gr.Button(
                        "Load uid",
                        variant="secondary",
                        elem_id="uid-load-btn",
                        size="lg",
                        scale=1,
                        min_width=120,
                    )

                mix_lm = gr.Audio(
                    label="Mixture (from wav.scp)",
                    type="filepath",
                    interactive=False,
                    sources=[],
                )
                kw_hint = gr.Markdown(
                    "_Load a uid to show official keyword hints._")
                with gr.Row():
                    s1_text = gr.Textbox(
                        label="Speaker 1 — full transcript (copy from here)",
                        lines=4,
                        interactive=True,
                    )
                    s2_text = gr.Textbox(
                        label="Speaker 2 — full transcript (copy from here)",
                        lines=4,
                        interactive=True,
                    )
                keyword_lm = gr.Textbox(
                    label="Keyword (enroll text, English)",
                    lines=2,
                    placeholder="Paste a consecutive English phrase from a transcript above",
                )
                extract_lm = gr.Button("Extract", variant="primary")
                out_lm = gr.Audio(
                    type="filepath",
                    label="Extracted speaker",
                    interactive=False,
                    sources=[],
                )
                heat_lm = gr.Image(
                    type="filepath",
                    label="Attention heatmap (time-aligned, seconds)",
                )

                load_btn.click(
                    fn=lookup_uid,
                    inputs=uid_box,
                    outputs=[mix_lm, s1_text, s2_text, kw_hint, keyword_lm],
                )
                extract_lm.click(
                    fn=run_extract,
                    inputs=[keyword_lm, mix_lm],
                    outputs=[out_lm, heat_lm],
                )

            with gr.Tab("OOD upload"):
                gr.Markdown(
                    "**English only.** Upload a mixture and type an **English** "
                    "target keyword yourself. Other languages are unsupported "
                    "and not recommended. No transcript lookup. "
                    "Quality outside LibriMix is **not guaranteed**."
                )
                keyword_ood = gr.Textbox(
                    label="Keyword (enroll text, English)",
                    lines=2,
                    placeholder="Type an English keyword / phrase yourself",
                )
                mix_ood = gr.Audio(
                    label="Mixture (upload)",
                    type="numpy",
                    sources=["upload"],
                )
                extract_ood = gr.Button("Extract", variant="primary")
                out_ood = gr.Audio(
                    type="filepath",
                    label="Extracted speaker",
                    interactive=False,
                    sources=[],
                )
                heat_ood = gr.Image(
                    type="filepath",
                    label="Attention heatmap (time-aligned, seconds)",
                )
                extract_ood.click(
                    fn=run_extract,
                    inputs=[keyword_ood, mix_ood],
                    outputs=[out_ood, heat_ood],
                )

        # Prefill default LibriMix uid on startup when available
        if DEFAULT_UID in index.entries:
            demo.load(
                fn=lookup_uid,
                inputs=uid_box,
                outputs=[mix_lm, s1_text, s2_text, kw_hint, keyword_lm],
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="DAE-TSE local Gradio demo")
    parser.add_argument("--exp_dir", default=None, help="Backbone exp dir")
    parser.add_argument(
        "--wav_scp",
        default=os.environ.get("DAE_TSE_WAV_SCP", DEFAULT_WAV_SCP),
        help="LibriMix test wav.scp (uid mix [s1] [s2])",
    )
    parser.add_argument(
        "--text_enroll",
        default=os.environ.get("DAE_TSE_TEXT_ENROLL", DEFAULT_JSONL),
        help="Keyword jsonl with normalized_text per s1|s2/uid",
    )
    parser.add_argument("--server_name", default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    exp_dir = os.path.realpath(args.exp_dir) if args.exp_dir else resolve_exp_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = LibriMixIndex.load(args.wav_scp, args.text_enroll)
    if not index.entries:
        print(
            "[demo] WARNING: empty LibriMix index. "
            "Prepare data/clean/test/wav.scp (see infer.sh stage 1) "
            "or pass --wav_scp / --text_enroll. OOD tab still works."
        )

    print(f"Loading model from {exp_dir} on {device} ...")
    model = load_model(exp_dir).to(device).eval()
    converter = build_text2phoneme()
    app = build_app(model, converter, device, index)

    # Allow original LibriMix dirs if some paths are returned directly.
    allowed = {INPUT_DIR, OUTPUT_DIR}
    for entry in index.entries.values():
        allowed.add(os.path.dirname(entry.mix_path))
        if len(allowed) > 8:
            break

    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=sorted(allowed),
        css="""
        #uid-load-row { align-items: stretch; }
        #uid-load-col { justify-content: flex-end; }
        #uid-load-col .uid-load-label {
          display: block;
          font-size: var(--text-md);
          line-height: var(--line-sm);
          margin-bottom: var(--spacing-sm);
          color: transparent;
          user-select: none;
          height: 1.2em;
        }
        #uid-load-btn { width: 100%; height: 100%; min-height: 42px; }
        """,
    )


if __name__ == "__main__":
    main()
