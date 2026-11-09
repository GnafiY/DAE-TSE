# ref: wenet dataset.py
#from multiprocessing.dummy import Process
import copy
import random
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

from kce.utils import read_list
from kce.data.factory import (
    process_raw,
    process_corruption,
    process_speech_feats,
    process_fix_keyword,
    process_text_feats,
    process_sampled_keyword_from_label,
    process_list_data,
    make_length,
    make_batch,
    process_speaker_label,
    fetch_tensor,
)

class Processer(IterableDataset):
    def __init__(
        self,
        source,
        f,
        *args,
        **kw
    ):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        self.source.set_epoch(epoch)

    def __iter__(self):
        assert self.source is not None
        return self.f((iter(self.source)), *self.args, **self.kw)

    #def apply(self, f):
    #    return Processer(self, f, *self.args, **self.kw)

class DistributedSampler:
    def __init__(
        self,
        shuffle=True,
        partition=True
    ):
        self.epoch=-1
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
            rank = self.rank,
            world_size = self.world_size,
            worker_id = self.worker_id,
            num_workers=self.num_workers
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, indexes):
        if self.partition:
            if self.shuffle:
                random.Random(self.epoch).shuffle(indexes)
            indexes = indexes[self.rank::self.world_size]       # indexes are splitted into self.world_size parts
        indexes = indexes[self.worker_id::self.num_workers]     # indexes are splitted into self.num_workers parts
        return indexes

class DataList(IterableDataset):
    def __init__(
        self,
        lists: list[str],   # jsonl format
        self_corruption: bool = False,
        none_target_corruption_list=None,
        rirs_list=None,
        shuffle=True,
        partition=True,
    ):
        self.lists = lists
        self.sampler = DistributedSampler(shuffle, partition)
        self.self_corruption = self_corruption
        if none_target_corruption_list is not None:
            self.none_target_corruption_list = none_target_corruption_list
            self.num_none_target_corruption = len(self.none_target_corruption_list)
            self.none_target_corruption = True
        else:
            self.none_target_corruption = False
        if rirs_list is not None:
            self.rirs_list = rirs_list
            self.reverb = True
        else:
            self.reverb = False

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def padding(self, org_list, target_len):
        org_len = len(org_list)
        assert org_len < target_len
        num_repeat = target_len // org_len
        new_list = copy.deepcopy(org_list)
        for x in range(num_repeat):
            new_list += org_list
        return new_list

    def make_corrupt_candidate(self, lists, indexes, num_candidate=5):
        idx = random.choices(indexes, k=num_candidate)
        candidate = []
        for i in idx:
            candidate.append(lists[i])
        if num_candidate == 1:
            candidate = candidate[0]
        return candidate

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = list(range(len(self.lists)))
        indexes = self.sampler.sample(indexes)
        if self.self_corruption:
            self_corrupt_lists = copy.deepcopy(self.lists)
            self_corrupt_indexes = list(range(len(self_corrupt_lists)))
            self_corrupt_indexes = self.sampler.sample(self_corrupt_indexes)
        if self.none_target_corruption:
            none_target_corrupt_lists = self.none_target_corruption_list
            none_target_corrupt_indexes = list(range(len(none_target_corrupt_lists)))
            none_target_corrupt_indexes = self.sampler.sample(none_target_corrupt_indexes)
        if self.reverb:
            rirs_lists = self.rirs_list
            rirs_indexes = list(range(len(rirs_lists)))
            rirs_indexes = self.sampler.sample(rirs_indexes)

        for i, index in enumerate(indexes):
            data = dict(src=self.lists[index], epoch=self.sampler.epoch)
            data.update(sampler_info)
            if self.self_corruption:
                self_corrupt_candidate = self.make_corrupt_candidate(
                    self_corrupt_lists, self_corrupt_indexes, num_candidate=150
                )
                data.update(self_corruption=copy.deepcopy(self_corrupt_candidate))

                one_neg_candidate = self.make_corrupt_candidate(
                    self_corrupt_lists, self_corrupt_indexes, num_candidate=150
                )
                data.update(neg_candidate=one_neg_candidate)
            if self.none_target_corruption:
                none_target_corrupt_candidate = self.make_corrupt_candidate(
                    none_target_corrupt_lists, none_target_corrupt_indexes
                )
                data.update(none_target_corruption=none_target_corrupt_candidate)
            if self.reverb:
                rirs_src = self.make_corrupt_candidate(rirs_lists, rirs_indexes, num_candidate=1)
                data.update(rirs=rirs_src)
            yield data


def Dataset(
    conf: dict,
    d_list,
):
    # check speech config
    sph_config: dict = conf.get('sph_config', None)
    if not sph_config:
        raise NotImplementedError(
            "sph_config should be specific, there are no any default config for " +
            "speech feats"
        )
    data_list_config: dict = {'lists': d_list}
    shuffle: bool = conf.get('shuffle', True)
    data_list_config.update({'shuffle': shuffle})
    corruption_config = {}
    if sph_config.get('self_corruption', False):
        corruption_config.update({'self_corruption': sph_config.get('self_corruption')})
        self_corruption = True
        data_list_config.update({'self_corruption': self_corruption})

    if sph_config.get('none_target_corruption', False):
        corruption_config.update({'none_target_corruption': sph_config.get('none_target_corruption')})
        none_target_corruption_list = corruption_config['none_target_corruption']['corrupt_list']
        none_target_corruption_list = read_list(none_target_corruption_list)
        data_list_config.update({'none_target_corruption_list': none_target_corruption_list})
    if sph_config.get('rirs_list', False):
        rirs_list = sph_config['rirs_list']
        rirs_list = read_list(rirs_list)
        data_list_config.update({'rirs_list': rirs_list})

    sv_config = conf.get('sv_config', None)

    # Build data list
    dataset = DataList(**data_list_config)

    # START DATA PROCESS!!!
    dataset = Processer(dataset, process_raw)

    if len(corruption_config) > 0:
        dataset = Processer(dataset, process_corruption, corruption_config)

    # prepare speech feats
    dataset = Processer(dataset, process_speech_feats, sph_config)

    # prepare text feats, such as label, keyword etc.
    dataset = Processer(dataset, process_text_feats)

    # process keyword setting
    keyword_setting = conf.get('keyword_config', None)
    if isinstance(keyword_setting, dict):
        keyword_format = keyword_setting.get('format')
        assert keyword_format in ['sample', 'fix']
        keyword_config: dict = keyword_setting.get('config', {})
        if keyword_format == 'sample':
            crpt_list = copy.deepcopy(d_list)
            random.shuffle(crpt_list)
            #keyword_config.update({'neg_list': crpt_list, 'neg_len': len(d_list)})
            keyword_config.update({'neg_len': 70})
            dataset = Processer(dataset, process_sampled_keyword_from_label, **keyword_config)
        elif keyword_format == 'fix':
            dataset = Processer(dataset, process_fix_keyword, **keyword_config)
        else:
            raise NotImplementedError("keyword format only Support <sample> / <fix_with_segment>")
    elif keyword_setting is not None:
        raise NotImplementedError("keyword config should be dict")
    else:
        pass

    # process list data
    dataset = Processer(dataset, process_list_data)

    # process length information
    dataset = Processer(dataset, make_length)

    # read speaker label
    if sv_config:
        dataset = Processer(dataset, process_speaker_label, **sv_config)

    # make batch
    dataset = Processer(dataset, make_batch, conf.get('batch_size', 256))

    # if fetch_key from config is not None, use that fetch_key
    # fetch tensor
    fetch_key: list[str] = conf.get('fetch_key', 'speech,label,keyword').split(",")
    dataset = Processer(dataset, fetch_tensor, fetch_key)

    return dataset
