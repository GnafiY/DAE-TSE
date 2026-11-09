# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Shuai Wang (wsstriving@gmail.com)
# Slimmed for DAE-TSE inference (raw LibriMix + keyword cues).

import copy
import json
import logging
import random

import numpy as np
import torch
import torchaudio


def parse_raw(data):
    """Parse wav.scp lines into mixture / source waveforms.

    Expected scp format per line:
      key mix.wav s1.wav s2.wav
    """
    for sample in data:
        assert "src" in sample
        try:
            key, wav_file, *single_spk_wavs = sample["src"].split()
            waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(
                key=key,
                wav_mix=waveform,
                num_speaker=len(single_spk_wavs),
                sample_rate=sample_rate,
            )
            spk_uids = [_.split("-")[0] for _ in key.split("_")]
            for spk_idx, single_spk_wav in enumerate(single_spk_wavs):
                waveform, sample_rate = torchaudio.load(single_spk_wav)
                example[f"spk{spk_idx + 1}"] = spk_uids[spk_idx]
                example[f"wav_spk{spk_idx + 1}"] = waveform
            yield example
        except Exception:
            logging.warning("Failed to read {}".format(sample.get("src")))


def resample(data, resample_rate=16000):
    """Resample waveforms to ``resample_rate`` (inplace)."""
    for sample in data:
        sample_rate = sample["sample_rate"]
        if sample_rate != resample_rate:
            all_keys = list(sample.keys())
            sample["sample_rate"] = resample_rate
            for key in all_keys:
                if key.startswith("wav"):
                    sample[key] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=resample_rate)(
                            sample[key])
        yield sample


def add_dummy_spk_embeds(data):
    """Placeholder enrollment tensors for collate (DAE-TSE discards enroll)."""
    for sample in data:
        for i in range(sample["num_speaker"]):
            sample[f"embed_spk{i + 1}"] = np.zeros((1, 1), dtype=np.float32)
        yield sample


def load_additional(data, state, additional_conf: dict = {}):
    """Attach keyword phoneme cues (TC-ASR) for DAE-TSE."""
    if additional_conf == {}:
        for sample in data:
            yield sample
        return

    assert "data_type" in additional_conf
    data_type = additional_conf["data_type"]
    if data_type != "TC-ASR":
        raise NotImplementedError(f"Unsupported additional data_type: {data_type}")

    KEPT_KEYS = ["key", "kw_candidate", "phn_label"]
    additional_src = {}
    with open(additional_conf["data_list"]) as f:
        for line in f.readlines():
            line = json.loads(line)
            additional_src[line["key"]] = {
                key: line.get(key, None)
                for key in KEPT_KEYS
            }

    for sample in data:
        key = sample["key"]
        additional_dict = {"data_type": data_type, "data": []}
        for spk_idx, _uid in enumerate(key.split("_")):
            text_uid = f"s{spk_idx + 1}/{key}"
            additional = copy.deepcopy(additional_src[text_uid])
            label = additional["phn_label"]
            kw_candidate = additional.get("kw_candidate", None)
            kw_sampler = additional_conf.get("kw_sampler", "fix")

            if kw_sampler == "random":
                a, b = additional_conf.get("kw_len_range", [2, 6])
                kw, kw_pos = sample_kw_from_label(
                    label, kw_candidate, kw_sampler, a, b)
            elif kw_sampler == "fix":
                assert isinstance(kw_candidate, list) and len(kw_candidate) > 0
                kw, kw_pos = sample_kw_from_label(
                    label, kw_candidate, kw_sampler)
            else:
                raise NotImplementedError(f"Unknown kw_sampler: {kw_sampler}")

            kw = [71, kw, 72]  # psok / peok special tokens
            additional.update({"kw": kw, "kw_pos": kw_pos})
            additional_dict["data"].append(additional)

        sample["additional"] = additional_dict
        yield sample


def flatten_list(nasted_list: list) -> list:
    flattened = []
    for elem in nasted_list:
        if isinstance(elem, list):
            flattened.extend(flatten_list(elem))
        else:
            flattened.append(elem)
    return flattened


def sample_kw_from_label(label, kw_candidate=None, kw_sampler="random", a=2, b=6):
    if kw_sampler == "random":
        kw_len = random.randint(a, b)
        if kw_candidate:
            kw_len = kw_len if kw_len < len(kw_candidate) else 1
            kw_pos_idx = (random.randint(0, len(kw_candidate) - kw_len - 1)
                          if len(kw_candidate) > kw_len + 1 else 0)
            kw_pos = kw_candidate[kw_pos_idx]
            if kw_pos_idx + kw_len >= len(kw_candidate):
                kw_len -= 1
            kw_len = kw_candidate[kw_pos_idx + kw_len] - kw_pos
        else:
            kw_pos = (random.randint(0, len(label) - kw_len)
                      if len(label) > kw_len else 0)
        kw = label[kw_pos:kw_pos + kw_len]
    else:  # fixed (inference)
        kw = [label[_] for _ in kw_candidate]
        kw_pos = kw_candidate[0]
    return kw, kw_pos


def shuffle(data, shuffle_size=2500):
    """Local shuffle the data

    Args:
        data: Iterable[{key, wavs, spks}]
        shuffle_size: buffer size for shuffle

    Returns:
        Iterable[{key, wavs, spks}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x
