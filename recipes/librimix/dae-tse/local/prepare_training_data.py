"""Prepare raw Libri2Mix lists for the migrated whole-utterance trainer."""

import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix_data_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/clean"))
    args = parser.parse_args()
    for subset in ("train-100", "dev", "test"):
        folder = (args.mix_data_path / subset).resolve()
        paths = sorted((folder / "mix_clean").glob("*.wav"))
        if not paths:
            raise FileNotFoundError(folder / "mix_clean")
        output = args.output / subset
        output.mkdir(parents=True, exist_ok=True)
        scp, single, utt2spk, single_utt2spk = [], [], [], []
        enroll = [[], []]
        speakers = {}
        for mixture in paths:
            key = mixture.stem
            sources = [folder / f"s{i}" / mixture.name for i in (1, 2)]
            if not all(source.is_file() for source in sources):
                raise FileNotFoundError(f"Missing source for {mixture}")
            identities = [part.split("-")[0] for part in key.split("_")]
            scp.append(f"{key} {mixture} {sources[0]} {sources[1]}\n")
            utt2spk.append(f"{key} {' '.join(identities)}\n")
            for i, (source, speaker) in enumerate(zip(sources, identities)):
                name = f"s{i + 1}/{key}"
                single.append(f"{name} {source}\n")
                single_utt2spk.append(f"{name} {speaker}\n")
                enroll[i].append(f"{key} {source}\n")
                speakers.setdefault(speaker, []).append(str(source))
        content = {"wav.scp": scp, "single.wav.scp": single,
                   "utt2spk": utt2spk, "single.utt2spk": single_utt2spk,
                   "spk1.enroll": enroll[0], "spk2.enroll": enroll[1]}
        for name, lines in content.items():
            (output / name).write_text("".join(lines))
        (output / "spk2enroll.json").write_text(json.dumps(speakers, indent=2))
        print(f"{subset}: {len(paths)} whole mixtures", flush=True)
