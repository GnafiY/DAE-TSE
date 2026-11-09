import argparse
import json

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate speaker label json file from train and dev datalist files"
    )
    parser.add_argument(
        "--train_datalist",
        type=str,
        help="Path to the training datalist file (jsonl format)",
        required=True,
    )
    parser.add_argument(
        "--valid_datalist",
        type=str,
        help="Path to the development datalist file (jsonl format)",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to the output speaker label json file",
        default="data/dump/speaker_label",
    )
    return parser.parse_args()

def get_speaker_id(utt_id):
    # for LibriSpeech, speaker ID is the first part of the utterance ID split by '-'
    return utt_id.split('-')[0]

if __name__ == "__main__":
    args = get_args()

    input_lines = []
    for input_file in [args.train_datalist, args.valid_datalist]:
        with open(input_file, 'r') as f:
            input_lines += f.readlines()

    speaker_list = []
    for line in input_lines:
        data = json.loads(line)
        utt_id = data['key']
        speaker_id = get_speaker_id(utt_id)
        speaker_list.append(speaker_id)

    speaker_set = set(speaker_list)
    speaker_dict = {speaker_id: idx for idx, speaker_id in enumerate(sorted(speaker_set))}

    output_path = f"{args.output_dir}/spk2cls.json"
    with open(output_path, 'w') as f:
        json.dump(speaker_dict, f, indent=4)
