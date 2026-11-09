import argparse
import json
import os

import numpy as np

SUCCESS_THRESHOLD_FOR_SISNRi = 1.0


def get_mean_std(results: list[float]) -> str:
    if not results:
        return "n/a"
    return f"{np.mean(results):.2f}±{np.std(results):.2f}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_or_dir', required=True, type=str)
    parser.add_argument(
        '--success_threshold_for_sisnri',
        type=float,
        default=SUCCESS_THRESHOLD_FOR_SISNRi,
        help='SI-SNRi >= threshold counts as a successful case (default: 1.0).',
    )
    return parser.parse_args()


def merge_jsonl(args):
    input_lines = []
    if args.jsonl_or_dir.endswith('.jsonl'):
        input_lines = open(args.jsonl_or_dir).readlines()
    else:
        for fname in os.listdir(args.jsonl_or_dir):
            if fname.endswith('.jsonl'):
                with open(os.path.join(args.jsonl_or_dir, fname)) as f:
                    input_lines.extend(f.readlines())

    thr = args.success_threshold_for_sisnri
    sisnri_all = []
    sisnri_success = []
    sisnri_fail = []

    for line in input_lines:
        result = json.loads(line)
        sisnri = result['SI-SNRi']
        sisnri_all.append(sisnri)
        if sisnri >= thr:
            sisnri_success.append(sisnri)
        else:
            sisnri_fail.append(sisnri)

    n = len(sisnri_all)
    n_ok = len(sisnri_success)
    n_fail = len(sisnri_fail)
    print(f"Cases (success_threshold_for_sisnr={thr}): "
          f"success={n_ok} ({100.0 * n_ok / n:.2f}%), "
          f"fail={n_fail} ({100.0 * n_fail / n:.2f}%)")
    print(f"SI-SNRi [successful]:   {get_mean_std(sisnri_success)}")
    print(f"SI-SNRi [unsuccessful]: {get_mean_std(sisnri_fail)}")
    print(f"SI-SNRi [overall]:      {get_mean_std(sisnri_all)}")


if __name__ == "__main__":
    args = get_args()
    merge_jsonl(args)
