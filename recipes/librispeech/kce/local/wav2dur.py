#!/usr/bin/env python3
# Source: https://github.com/GnafiY/wekws/blob/38e88b74e52e5fa225a9641f32722d0bf4b2637b/tools/wav2dur.py
# Adapted for this recipe; see repository LICENSE.
"""Adapted from WeKws tools/wav2dur.py: calculate one SCP shard."""

import sys

import torch
import torchaudio


def main():
    torch.set_num_threads(1)
    scp, output = sys.argv[1:]
    count = 0
    total_duration = 0.0
    with open(scp) as source, open(output, "w") as target:
        for line in source:
            key, path = line.strip().split(maxsplit=1)
            waveform, sample_rate = torchaudio.load(path)
            duration = waveform.shape[-1] / float(sample_rate)
            target.write(f"{key} {duration:.2f}\n")
            count += 1
            total_duration += duration
    print(f"Processed {count} utterances; total {total_duration:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
