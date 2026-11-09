import nltk


import argparse
import g2p_en
import glob
import os
import json

from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)

TEXT_SUFFIX = ".trans.txt"

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
        help='Output directory for text label files',
        default='data/dump/text_label',
    )
    parser.add_argument(
        '--p2idx_path',
        type=str,
        default='data/phoneme2int.txt',
        help='Path to the phoneme to index mapping file',
    )
    parser.add_argument(
        '--lexicon_path',
        type=str,
        default='data/word2lexicon.txt',
        help='Path to the lexicon file that maps words to phoneme index sequences (for speedup)',
    )
    return parser.parse_args()

def read_word2lexicon(file):
    lines = open(file).readlines()
    lines = [_.strip().split(maxsplit=1) for _ in lines]
    lexicon = {}
    for line in lines:
        word, ints = line
        ints = [int(_) for _ in ints.split()]
        lexicon[word] = ints
    return lexicon

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory created: {args.output_dir}")

    g2p_obj = None
    p2idx = {}
    if os.path.exists(args.p2idx_path):
        CUSTOM_P2IDX_PATH = args.p2idx_path
        def read_p2idx(file):
            lines = open(file).readlines()
            lines = [_.strip().split() for _ in lines]
            p2idx = {}
            for line in lines:
                p, idx = line
                p2idx[p] = int(idx)
            return p2idx
        p2idx = read_p2idx(CUSTOM_P2IDX_PATH)

    word2lexicon_mapping = read_word2lexicon(args.lexicon_path) if os.path.exists(args.lexicon_path) else {}
    if word2lexicon_mapping == {}:
        logging.warning("No lexicon file found, will generate lexicon on the fly.")
    else:
        logging.warning(f"Loaded lexicon from {args.lexicon_path}, {len(word2lexicon_mapping)} entries.")

    input_lines = []
    text_pattern = "{}/{}/**/*{}".format(args.librispeech_dir, args.subset, TEXT_SUFFIX)
    text_files = glob.glob(text_pattern, recursive=True)
    for text_file in text_files:
        lines = open(text_file).readlines()
        input_lines.extend(lines)

    def word2lexicon(word):
        word = word.lower()
        if word in word2lexicon_mapping:
            return word2lexicon_mapping[word]
        else:
            global g2p_obj
            if g2p_obj is None:
                g2p_obj = g2p_en.G2p()
            phones = g2p_obj(word)
            word2lexicon_mapping[word] = [p2idx[p] for p in phones if p in p2idx]
            return word2lexicon_mapping[word]

    output_lines = []
    for line in tqdm(input_lines):
        uid, words = line.strip().split(maxsplit=1)
        lexicons = []
        for word in words.split():
            lexicons.append(word2lexicon(word))
        output_lines.append(json.dumps({
            'key': uid,
            'phn_label': lexicons,
            'text': words,
        }) + '\n')

    output_path = os.path.join(args.output_dir, f"{args.subset}.jsonl")
    with open(output_path, 'w') as f:
        f.writelines(output_lines)
    logging.info(f"Text label file generated at: {output_path}")
