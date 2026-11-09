# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Shuai Wang (wsstriving@gmail.com)
# DAE-TSE raw whole-utterance training and inference.

import random

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

import wesep.dataset.processor as processor
from wesep.dataset.processor import flatten_list
from wesep.utils.file_utils import read_lists


class Processor(IterableDataset):

    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)


class DistributedSampler:

    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition

    def update(self):
        assert dist.is_available()
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        return dict(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        data = list(range(len(data)))
        if len(data) <= self.num_workers:
            if self.shuffle:
                random.Random(self.epoch).shuffle(data)
        else:
            if self.partition:
                if self.shuffle:
                    random.Random(self.epoch).shuffle(data)
                data = data[self.rank::self.world_size]
            data = data[self.worker_id::self.num_workers]
        return data


class DataList(IterableDataset):

    def __init__(self, lists, shuffle=False, partition=True, repeat_dataset=False):
        self.lists = lists
        self.repeat_dataset = repeat_dataset
        self.sampler = DistributedSampler(shuffle, partition)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        while indexes:
            for index in indexes:
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data
            if not self.repeat_dataset:
                break


def tse_collate_fn(batch, mode="max"):
    new_batch = {}
    wav_mix = []
    wav_targets = []
    spk_embeds = []
    spk = []
    key = []

    for s in batch:
        for i in range(s["num_speaker"]):
            wav_mix.append(s["wav_mix"])
            wav_targets.append(s["wav_spk{}".format(i + 1)])
            spk.append(s["spk{}".format(i + 1)])
            key.append(s["key"])
            spk_embeds.append(
                torch.from_numpy(s["embed_spk{}".format(i + 1)].copy()))

    new_batch["wav_mix"] = torch.concat(wav_mix)
    new_batch["wav_targets"] = torch.concat(wav_targets)
    new_batch["spk_embeds"] = torch.concat(spk_embeds)
    new_batch["spk"] = spk
    new_batch["key"] = key
    new_batch["additional"] = tse_collate_additional_fn(batch)
    new_batch["spk_label"] = torch.zeros(len(wav_mix), dtype=torch.long)
    return new_batch


def tse_collate_additional_fn(batch):
    if "additional" not in batch[0]:
        return {}

    data_type = batch[0]["additional"]["data_type"]
    if data_type != "TC-ASR":
        raise NotImplementedError(f"Unsupported additional data_type: {data_type}")

    kw_labels = []
    kw_lens = []
    for s in batch:
        for i in range(s["num_speaker"]):
            kw_label = flatten_list(s["additional"]["data"][i]["kw"])
            kw_labels.append(torch.tensor(kw_label))
            kw_lens.append(len(kw_label))

    return {
        "data_type": data_type,
        "data": {
            "kw_label": pad_sequence(kw_labels, batch_first=True),
            "kw_len": torch.tensor(kw_lens, dtype=torch.long),
        },
    }


def Dataset(
    data_type,
    data_list_file,
    configs,
    spk2embed_dict=None,
    spk1_embed=None,
    spk2_embed=None,
    state="test",
    joint_training=True,
    whole_utt=True,
    repeat_dataset=False,
    additional_conf=None,
    **_unused,
):
    """Build the DAE-TSE whole-utterance dataset (raw wav.scp + keyword JSONL)."""
    assert data_type == "raw", "This recipe supports raw wav.scp"
    assert state in ("train", "val", "test")
    assert whole_utt, "This recipe supports whole-utterance training only"
    additional_conf = additional_conf or {}

    lists = read_lists(data_list_file)
    dataset = DataList(lists, shuffle=state == "train",
                       repeat_dataset=repeat_dataset)
    dataset = Processor(dataset, processor.parse_raw)

    if state == "train" and configs.get("shuffle", False):
        dataset = Processor(dataset, processor.shuffle,
                            **configs.get("shuffle_args", {}))

    resample_rate = configs.get("resample_rate", 16000)
    dataset = Processor(dataset, processor.resample, resample_rate)
    # Enrollment audio is unused by DAE-TSE (cue comes from mixture + keywords).
    dataset = Processor(dataset, processor.add_dummy_spk_embeds)
    dataset = Processor(dataset, processor.load_additional, state, additional_conf)
    return dataset
