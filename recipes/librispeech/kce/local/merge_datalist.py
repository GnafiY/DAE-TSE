import argparse
import json
import os
import logging
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subsets',
        type=str,
        required=True,
        help='Dataset subsets to process',
        choices=['train-other-500', 'train-clean-360', 'train-clean-100', 'test-clean', 'test-other', 'dev-clean', 'dev-other'],
        nargs='+',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for shuffling',
    )
    parser.add_argument(
        '--datalist_dir',
        type=str,
        default='data/dump/datalist',
        help='Directory containing datalist files',
    )
    parser.add_argument(
        '--duration_dir',
        type=str,
        default='data/dump/duration',
        help='Directory containing duration files',
    )
    parser.add_argument(
        '--max_duration',
        type=float,
        default=20.0,
        help='Maximum duration (in seconds) to filter utterances',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output datalist file path',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=-1,
        help='Number of samples to include in the output (for validation set truncation), non-positive means all samples',
    )
    return parser.parse_args()

def read_datalist(file) -> dict:
    lines = open(file).readlines()
    ret = {}
    for line in lines:
        entry = json.loads(line.strip())
        ret[entry['key']] = entry
    return ret

def read_duration(file) -> dict:
    lines = open(file).readlines()
    ret = {}
    for line in lines:
        uid, dur = line.strip().split()
        ret[uid] = float(dur)
    return ret

if __name__ == '__main__':
    args = get_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    logging.basicConfig(
        filename=args.output_file.split('.')[0] + '.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Output directory created: {os.path.dirname(args.output_file)}")

    duration_meta: dict = {}
    for subset in args.subsets:
        duration_file = os.path.join(args.duration_dir, f'{subset}.dur')
        duration_meta.update(read_duration(duration_file))

    datalist_meta: dict = {}
    for subset in args.subsets:
        datalist_file = os.path.join(args.datalist_dir, f'{subset}.jsonl')
        datalist_meta.update(read_datalist(datalist_file))

    assert set(duration_meta.keys()) == set(datalist_meta.keys()), "Mismatch between duration and datalist keys"

    if args.num_samples <= 0:
        random.seed(args.seed)
        random.shuffle(list(datalist_meta.keys()))

    output_lines = []
    filter_count = 0
    for key, duration in duration_meta.items():
        if duration > args.max_duration:
            filter_count += 1
            continue


        # limit the number of samples if specified (for validation set truncation)
        if args.num_samples > 0 and len(output_lines) >= args.num_samples:
            break

        entry = datalist_meta[key]
        output_lines.append(json.dumps(entry) + '\n')

    with open(args.output_file, 'w') as fout:
        fout.writelines(output_lines)

    logging.info(f"Written {len(output_lines)} entries to {args.output_file}")
    logging.info(f"Filtered out {filter_count} entries exceeding max duration of {args.max_duration} seconds")
