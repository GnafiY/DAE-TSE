#!/usr/bin/env python3
"""Convert English text to phoneme-id sequences used by DAE-TSE keyword cues.

Example:
  python local/text2phoneme.py "Hey Siri open the door"
  python local/text2phoneme.py --text "hello world" --show_phones
  python local/text2phoneme.py --input phrases.txt --output cues.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from typing import Dict, List, Optional

# ASCII punct stripped by translate; unicode punct via category below.
# Apostrophes are kept (e.g. infant's).
PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RESOURCE_DIR = os.path.realpath(
    os.path.join(_HERE, '..', 'data', 'text_cue'))


def normalize_text(text: str) -> str:
    """Lowercase, unify apostrophes, drop punctuation (ASCII + unicode)."""
    text = (text or "").strip()
    text = (text.replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u0060", "'").replace("\u00b4", "'"))
    # Replace punct with space so "hello,world" / "keswick—march" stay split.
    cleaned = []
    for ch in text:
        if ch == "'":
            cleaned.append(ch)
        elif ch in PUNCTUATION or unicodedata.category(ch).startswith("P"):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return " ".join("".join(cleaned).lower().split())


def prepare_keyword_text(text: str) -> str:
    """Normalize keyword text; require English letters only (demo/CLI gate).

    Punctuation is stripped. Remaining characters must be ``a-z``, spaces,
    or apostrophes. Raises ``ValueError`` on empty or non-English input.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Keyword / enroll text cannot be empty.")
    normalized = normalize_text(raw)
    if not normalized:
        raise ValueError(
            "Keyword is empty after removing punctuation. "
            "Please enter an English phrase.")
    bad = sorted({
        ch for ch in normalized
        if not (ch == " " or ch == "'" or ("a" <= ch <= "z"))
    })
    if bad:
        shown = ", ".join(repr(c) for c in bad[:8])
        raise ValueError(
            "Only English letters (a-z) are supported for keywords "
            f"(apostrophes allowed). Non-English character(s): {shown}")
    return normalized


def read_p2idx(path: str) -> Dict[str, int]:
    p2idx = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            p, idx = parts
            p2idx[p] = int(idx)
    return p2idx


def read_idx2p(p2idx: Dict[str, int]) -> Dict[int, str]:
    return {idx: p for p, idx in p2idx.items()}


def read_word2lexicon(path: str) -> Dict[str, List[int]]:
    lexicon = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            word, ints = parts
            lexicon[word] = [int(x) for x in ints.split()]
    return lexicon


class Text2Phoneme:
    def __init__(
        self,
        p2idx_path: str,
        lexicon_path: Optional[str] = None,
    ):
        import g2p_en

        self.g2p = g2p_en.G2p()
        self.p2idx = read_p2idx(p2idx_path)
        self.idx2p = read_idx2p(self.p2idx)
        self.lexicon = {}
        if lexicon_path and os.path.exists(lexicon_path):
            self.lexicon = read_word2lexicon(lexicon_path)

    def word_to_ids(self, word: str) -> List[int]:
        if word in self.lexicon:
            return self.lexicon[word]
        phones = self.g2p(word)
        ids = [self.p2idx[p] for p in phones if p in self.p2idx]
        self.lexicon[word] = ids
        return ids

    def convert(self, text: str) -> dict:
        normalized = normalize_text(text)
        words = normalized.split()
        phn_label = [self.word_to_ids(w) for w in words]
        phones = [[self.idx2p[i] for i in ids] for ids in phn_label]
        return {
            'text': text,
            'normalized_text': normalized,
            'words': words,
            'phn_label': phn_label,
            'phones': phones,
        }


def get_args():
    parser = argparse.ArgumentParser(
        description='Convert English text to DAE-TSE phoneme-id sequences.')
    parser.add_argument(
        'text_positional',
        nargs='?',
        default=None,
        help='English text to convert (alternative to --text).',
    )
    parser.add_argument('--text', type=str, default=None, help='English text.')
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input file: one utterance text per line, or a JSONL with a text field.',
    )
    parser.add_argument(
        '--text_key',
        type=str,
        default='text',
        help='JSONL field name when --input is JSONL (default: text).',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional output JSONL path. Default: print to stdout.',
    )
    parser.add_argument(
        '--p2idx_path',
        type=str,
        default=os.path.join(_DEFAULT_RESOURCE_DIR, 'phoneme2int.txt'),
        help='Phoneme-to-index mapping (must match KCE training).',
    )
    parser.add_argument(
        '--lexicon_path',
        type=str,
        default=os.path.join(_DEFAULT_RESOURCE_DIR, 'word2lexicon.txt'),
        help='Optional word->phoneme-id cache for speed.',
    )
    parser.add_argument(
        '--show_phones',
        action='store_true',
        help='Also print ARPAbet phone symbols.',
    )
    parser.add_argument(
        '--no_lexicon',
        action='store_true',
        help='Ignore lexicon cache and always run g2p_en.',
    )
    return parser.parse_args()


def iter_inputs(args):
    text = args.text if args.text is not None else args.text_positional
    if text is not None:
        yield {'text': text}
        return

    if args.input is None:
        if not sys.stdin.isatty():
            for line in sys.stdin:
                line = line.strip()
                if line:
                    yield {'text': line}
            return
        raise SystemExit(
            'Provide text via positional/--text, --input, or stdin.')

    if args.input.endswith('.jsonl'):
        with open(args.input) as f:
            for line in f:
                item = json.loads(line)
                yield item
        return

    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                yield {'text': line}


def main():
    args = get_args()
    lexicon_path = None if args.no_lexicon else args.lexicon_path
    converter = Text2Phoneme(args.p2idx_path, lexicon_path)

    outputs = []
    for item in iter_inputs(args):
        if 'text' not in item and args.text_key in item:
            raw = item[args.text_key]
        else:
            raw = item.get('text') or item.get('pred_text') or item.get(
                args.text_key)
        if raw is None:
            raise SystemExit(f'Missing text field in item: {item}')

        result = converter.convert(raw)
        # Keep useful passthrough fields when converting JSONL enrollments.
        if 'key' in item:
            result['key'] = item['key']
        if not args.show_phones:
            result.pop('phones', None)
        outputs.append(result)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.',
                    exist_ok=True)
        with open(args.output, 'w') as f:
            for result in outputs:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f'Wrote {len(outputs)} line(s) to {args.output}', file=sys.stderr)
    else:
        for result in outputs:
            print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
