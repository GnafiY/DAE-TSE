import argparse
import glob
import os

import logging
logging.basicConfig(level=logging.INFO)

WAV_SUFFIX = '.flac'

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
        '--librispeech_dir',
        type=str,
        default='/path/to/LibriSpeech',
        help='Path to the LibriSpeech dataset directory',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for wavscp file',
        default='data/dump/wavscp',
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory created: {args.output_dir}")

    output_lines = []
    wav_pattern = "{}/{}/**/*{}".format(args.librispeech_dir, args.subset, WAV_SUFFIX)
    wav_files = glob.glob(wav_pattern, recursive=True)
    for wav_file in wav_files:
        key = wav_file.split('/')[-1].replace(WAV_SUFFIX, '')
        output_lines.append(f"{key} {wav_file}\n")

    output_path = f"{args.output_dir}/{args.subset}.scp"
    with open(output_path, 'w') as f:
        f.writelines(output_lines)
    logging.info(f"wavscp file generated at: {output_path}")
