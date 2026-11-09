"""Download the existing paper model resources without a local workspace dependency."""

import argparse
from pathlib import Path
from urllib.request import urlretrieve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("exp/kce"))
    parser.add_argument("--transcript", type=Path,
                        default=Path("data/text_cue/transcript_nemo.jsonl"))
    parser.add_argument("--revision", default="9bbb532")
    args = parser.parse_args()
    files = [(f"kce/{name}", args.output / name)
             for name in ("epoch_149.pt", "model.yaml", "data.yaml")]
    files.append(("transcript_nemo/transcript_nemo.jsonl", args.transcript))
    for resource, target in files:
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        urlretrieve(f"https://huggingface.co/GnafiY/DAE-TSE/resolve/"
                    f"{args.revision}/{resource}", temporary)
        temporary.replace(target)
        print(f"Downloaded {target}", flush=True)
