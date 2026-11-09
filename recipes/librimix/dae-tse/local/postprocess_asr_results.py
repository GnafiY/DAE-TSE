import nltk


import argparse
import json
import os
import g2p_en
import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm

PRED_TEXT_KEY = 'pred_text'
SUBSET_KEY = 'subset'
AUDIO_FILEPATH_KEY = 'audio_filepath'

PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'    # avoid removing 's etc.

def normalize_text(text: str) -> str:
    text = text.strip()
    # Remove all punctuation
    text = text.translate(str.maketrans("", "", PUNCTUATION))
    # Convert all upper case to lower case
    text = text.lower()
    return text

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asr_transcript', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument(
        '--p2idx_path',
        type=str,
        default='data/text_cue/phoneme2int.txt',
        help='Path to the phoneme to index mapping file',
    )
    parser.add_argument(
        '--lexicon_path',
        type=str,
        default='data/text_cue/word2lexicon.txt',
        help='Path to the lexicon file that maps words to phoneme index sequences (for speedup)',
    )
    return parser.parse_args()

def read_jsonl(file) -> list[dict]:
    data = []
    with open(file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

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
    input_lines = read_jsonl(args.asr_transcript)

    all_subsets = set([_[SUBSET_KEY] for _ in input_lines])
    output_content = {subset: [] for subset in all_subsets}

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

    def word2lexicon(word):
        if word in word2lexicon_mapping:
            return word2lexicon_mapping[word]
        else:
            global g2p_obj
            if g2p_obj is None:
                g2p_obj = g2p_en.G2p()
            phones = g2p_obj(word)
            word2lexicon_mapping[word] = [p2idx[p] for p in phones if p in p2idx]
            return word2lexicon_mapping[word]

    for item in tqdm(input_lines):
        subset = item[SUBSET_KEY]

        text = item.get(PRED_TEXT_KEY, "")
        normalized_text = normalize_text(text)

        # The published transcript path identifies the source as s1/key or s2/key.
        uid = item[AUDIO_FILEPATH_KEY].split('/../')[-1].split('.')[0]

        phn_label = [word2lexicon(word) for word in normalized_text.split()]

        ready_item = {
            "key": uid,
            "phn_label": phn_label,
            "normalized_text": normalized_text,
            "text": text,
            "audio_filepath": item[AUDIO_FILEPATH_KEY],
        }

        output_content[subset].append(json.dumps(ready_item) + '\n')

    for subset, lines in output_content.items():
        output_path = os.path.join(args.output_dir, f"{subset}.jsonl")
        with open(output_path, 'w') as f:
            f.writelines(lines)
