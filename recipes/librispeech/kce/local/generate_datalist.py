import argparse
import json
import os

import logging
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subset',
        type=str,
        required=True,
        help='Dataset subset to process',
        choices=['train-other-500', 'train-clean-360', 'train-clean-100', 'test-clean', 'test-other', 'dev-clean', 'dev-other'],
    )
    parser.add_argument(
        '--wavscp_dir',
        type=str,
        default='data/dump/wavscp',
        help='Directory to save the generated wavscp files',
    )
    parser.add_argument(
        '--text_label_dir',
        type=str,
        default='data/dump/text_label',
        help='Directory to save the generated text label files',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for datalist files',
        default='data/dump/datalist',
    )
    return parser.parse_args()

def read_wavsp(wavscp_file) -> dict:
    wavscp = {}
    with open(wavscp_file, 'r') as f:
        for line in f:
            key, path = line.strip().split(maxsplit=1)
            wavscp[key] = path
    return wavscp

def read_text_label(text_label_file) -> dict:
    text_label = {}
    with open(text_label_file, 'r') as f:
        for line in f:
            meta = json.loads(line.strip())
            key = meta['key']
            text_label[key] = meta
    return text_label

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory created: {args.output_dir}")

    wavscp_file = os.path.join(args.wavscp_dir, f'{args.subset}.scp')
    wavscp = read_wavsp(wavscp_file)

    text_label_file = os.path.join(args.text_label_dir, f'{args.subset}.jsonl')
    text_label = read_text_label(text_label_file)

    assert set(wavscp.keys()) == set(text_label.keys()), "Mismatch between wavscp and text label keys"

    output_lines = []
    for key in wavscp.keys():
        phn_label = text_label[key]['phn_label']
        kw_candidate = [_ for _ in range(len(phn_label))]

        entry = {
            'key': key,
            'sph': wavscp[key],
            'phn_label': phn_label,
            'kw_candidate': kw_candidate,
            'text': text_label[key]['text'],
        }
        output_lines.append(json.dumps(entry) + '\n')

    output_file = os.path.join(args.output_dir, f'{args.subset}.jsonl')
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    logging.info(f"Datalist file generated at: {output_file}")
