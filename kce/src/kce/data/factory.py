# Reference: WeNet processor.py.
import re
import random
import torch
import torchaudio
import json
import copy

import kce.data.kaldi_io as kaldi_io
import torchaudio.compliance.kaldi as kaldi
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from scipy.io import wavfile
from scipy import signal
from typing import Generator


def spectrum(wav, n_fft=512, hop_length=160):
    extractor = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
    spec = extractor(wav)
    spec = spec.squeeze(0)
    return spec.transpose(0,1).contiguous()


def mel_padding_wav(waveform, window_size=400, window_shift=160):
    num_samples = waveform.size(0)
    reversed_waveform = torch.flip(waveform, [0])
    m = (num_samples + (window_shift // 2)) // window_shift
    pad = window_size // 2 - window_shift // 2
    pad_right = reversed_waveform[:pad]
    if pad > 0:
        # Reflect padding repeats edge samples here, unlike torch.nn.functional.pad.
        pad_left = reversed_waveform[-pad:]
        waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
    else:
        # Negative padding trims the waveform at the front.
        waveform = torch.cat((waveform[-pad:], pad_right), dim=0)
    return waveform.view(1,-1)


def get_mel_scale(n_mels=80, sample_rate=16000, f_min=0, f_max=None, n_stft=201, norm=None, mel_scale='htk'):
    f_max = sample_rate // 2
    filter_bank = torchaudio.functional.melscale_fbanks(
        n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_scale
    )
    return filter_bank


def mel_spectrum(
        waveform,
        win_length=400,
        hop_length=160,
        n_fft=400,
        win_fn='hamm',
        pad_mode='reflect',
        pow=2,
        center=False, onesided=True,
    ):
    waveform = mel_padding_wav(waveform.view(-1), win_length, hop_length)
    win_fn = torch.hamming_window
    window = win_fn(win_length)
    spec_f = torch.stft(
        input=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
        normalized=False,
        onesided=onesided,
        return_complex=True,
    )
    filter_bank = get_mel_scale(n_mels=80)
    spec_f = spec_f.abs().pow(pow)
    mel_spec = torch.matmul(spec_f.transpose(-1,-2), filter_bank).transpose(-1,-2)
    return mel_spec.squeeze(0).transpose(0,1)

NONE_TENSOR_KEY = [
    'wav', 'key', 'sph', 'corruption_material', 'segment', 'segment_idx',
    'n_scorrupt', 'n_ncorrupt', 'num_corrupt', 'rirs', 'neg_candidate',
    'corrupt', 'self_corruption', 'none_target_corruption', 'nframes',
    'text',
]
CTC_KEY = [
    'label', 'crpt_label',
    'phn_label', 'c_phn_label',
]

TEXT_SPEC_TOKEN = {
    'with_trans': None,
    'psok': None, 'peok': None, 'punk': None
}

INPUT_DATA_LOADER = {
    'raw': torchaudio.load, 'kaldi': kaldi_io.read_mat, 'torch': torch.load,
    'rm_sr': lambda x: x[0], 'copy': copy.deepcopy, 'empty': lambda x: x
}

FEATS_EXTRACTOR = {
    'mfcc': kaldi.mfcc, 'fbank': kaldi.fbank,
    'spectrum': spectrum, 'mel_spec': mel_spectrum
}

MFCC_DEFAULT_SETTING = {
    'num_mel_bins': 23, 'num_ceps': 13, 'frame_length': 25, 'frame_shift': 10,
    'energy_floor': 0.0, 'low_freq': 20
}
FBANK_DEFAULT_SETTING = {
    'num_mel_bins': 40, 'frame_length': 25, 'frame_shift': 10
}
SPECTRUM_DEFAULT_SETTING = {
    'window': torch.hann_window(400), 'normalized': False, 'pad': 0
}

RANDOM_FACTOR = {
    'beta':np.random.beta, 'uniform': np.random.uniform, 'int': np.random.randint, 'random': np.random.random
}

TRANSFORM_FACTOR ={
    'nptolist': lambda x: x.tolist() if isinstance(x, np.ndarray) else x
}

# Feature-frame units: 100 frames/s; waveform units: 16000 samples/s.
DITHER_RANGE = [10, 1600]

RE_PATTERN = {'space': re.compile(r" +"), 'dot': re.compile(r"\.")}


def read_list(list_file):
    d_list = []
    with open(list_file, encoding='utf-8') as lf:
        for line in lf.readlines():
            d_list.append(line.strip())
    lf.close()
    return d_list


def sym2float(sym_list):
    int_list = list(map(lambda x: float(x), sym_list.split(" ")))
    return int_list


def int2sym(int_list):
    if not isinstance(int_list, list):
        int_list = [int_list]
    int_list = [str(x) for x in int_list]
    return int_list


def sym2int(sym_list):
    int_list = list(map(lambda x: int(x), sym_list.split(" ")))
    return int_list


def tensor2str(t):
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    t = list(t)
    t = list(map(lambda x: str(x), t))
    return t


def save_wav(wav, names):
    if isinstance(names, list):
        names = "_".join(names)
    torchaudio.save(
        "{}.wav".format(names), wav, sample_rate=16000, encoding="PCM_S", bits_per_sample=16
    )


def splice_feats(
        feats: list[torch.Tensor],
        left_context: int,
        right_context: int,
        seq=False,
    ):
    frames, nmel = feats.size()
    l_padding = torch.ones_like(torch.rand(left_context, nmel))
    r_padding = torch.ones_like(torch.rand(right_context, nmel))
    l_padding *= feats[0]
    r_padding *= feats[-1]
    feats = torch.cat([l_padding, feats, r_padding], dim=0)
    if seq:
        return feats
    else:
        splice_v = []
        for i in range(left_context + right_context + 1):
            v = feats[i:frames + i]
            splice_v.append(v)
        feats = torch.cat([v for v in splice_v], dim=-1)
        return feats


def max_energy(wav, frame_length=400, hop_length=160, shift_type='mid'):
    wav = wav.view(-1) # Mono waveform: [1, num_samples].
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    energy = torch.zeros(num_frames, dtype=torch.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = wav[start:end]
        energy[i] = torch.sum(frame ** 2)

    idx_frame = torch.argmax(energy)
    dice_frame = random.randint(0,5)
    if dice_frame % 2 == 0:
        idx_frame = idx_frame - dice_frame
        idx_frame = idx_frame if idx_frame > 0 else 0
    else:
        idx_frame = idx_frame + dice_frame

    idx_sample = idx_frame * hop_length + frame_length
    return idx_sample


def unfold_list(lst):
    if not isinstance(lst, list):
        lst = [lst]
    l = _unfold_list(lst)
    trans = int if re.search(RE_PATTERN['dot'], l) == None else float
    l = [trans(i) for i in l.split(" ") if i !=""]
    return l


def _unfold_list(lst):
    new = ""
    for x in lst:
        if isinstance(x, list):
            x = _unfold_list(x)
        x = str(x)
        new = new + x + " "
    return new


def random_one(pools, pool_len):
    return (pools[random.randint(0, pool_len-1)])


def spec_augment(spec, num_t_mask=2, num_f_mask=2, max_t=20, max_f=10):
    assert isinstance(spec, torch.Tensor)
    aug_spec = spec.clone().detach()
    max_frames = aug_spec.size(0)
    max_freq = aug_spec.size(1)

    for i in range(num_t_mask):
        start = np.random.randint(0, max_frames - 1)
        length = np.random.randint(1, max_t)
        end = min(max_frames, start + length)
        aug_spec[start:end, :] = 0

    for i in range(num_f_mask):
        start = np.random.randint(0, max_freq - 1)
        length = np.random.randint(1, max_f)
        end = min(max_freq, start + length)
        aug_spec[:, start:end] = 0
    return aug_spec


def wav_augment(waveform, config, rirs=None):
    if config.get("volume", False):
        volume_sampler = config['volume']['sampler']
        volume_sampler_config = config['volume']['config']
        ratio = RANDOM_FACTOR[volume_sampler](**volume_sampler_config)
        waveform = ratio * waveform

    if config.get("white_niose", False):
        noise_config = config['white_noise']['config']
        gua = torch.normal(**noise_config, size=waveform.size())
        waveform = waveform + gua

    if rirs:
        rirs_prob = config.get('rirs_prob', 0.4)
        if random.uniform(0,1) < rirs_prob:
            _, rirs_src = wavfile.read(rirs)
            rirs_src = rirs_src / np.sqrt(np.sum(rirs_src**2))
            rirs_src = rirs_src.astype(np.float32)
            l = waveform.size(1)
            waveform = waveform[0].numpy()
            waveform = signal.convolve(waveform, rirs_src)[:l]
            waveform = torch.from_numpy(waveform)
            waveform = waveform.view(1,-1)
    return waveform

'''
    wavs: [Tensor, Tensor], spk1, spk2
'''


def make_mix_wav(
        wavs: list[torch.Tensor],
        ratios: list[float],
        skip_idx=None,
        mean_dur: str ='max',
        delay_type: str = 'fix_dur',
        delay_config: dict = {},
        major=False,
    ):
    if skip_idx is not None:
        skip_feats = wavs[-skip_idx:]
        skip_ratios = ratios[-skip_idx:]
        wavs = wavs[:-skip_idx]
        ratios = ratios[:-skip_idx]
    else:
        skip_feats, skip_ratios = [],[]

    # Compute source L2 norms before padding.
    rms = [w.norm(p=2).item() for w in wavs]
    rms = list(map(lambda x: x if x > 0.01 else 1, rms)) # Avoid division by a near-zero source norm.
    max_rms = max(rms)

    ovr_ratio = -1
    if delay_config.get('use', False):
        wavs, ovr_ratio = make_wav_delay(wavs, delay_type, delay_config)

    wav_size = [wav.size(1) for wav in wavs]
    if mean_dur == 'max':
        wav_len = max(wav_size) if max(wav_size) >= 16000 else 16000
    else:
        wav_len = wav_size[0]

    wavs = [
        padding_wav(wavs[i], length=wav_len) for i in range(len(wavs))
    ]

    if major:
        assert len(wavs) > 1
        if major == 1:
            ratios[0], ratios[1] = (ratios[1], ratios[0]) if ratios[0]<ratios[1] else (ratios[0], ratios[1])
        else:
            ratios[1], ratios[0] = (ratios[1], ratios[0]) if ratios[0]<ratios[1] else (ratios[0], ratios[1])

    wavs: list[torch.Tensor] = [wavs[i]*(max_rms/rms[i]) for i in range(len(rms))] # Equalize source energy before applying independently sampled ratios.
    scaled_wav: list[torch.Tensor] = [wavs[i]*ratios[i] for i in range(len(wavs))]
    mix_wav: torch.Tensor = sum(scaled_wav)

    if torch.max(mix_wav) > 1 or torch.min(mix_wav) < -1:
        ratios = [r/sum(ratios) for r in ratios]
        scaled_wav = [wavs[i]*ratios[i] for i in range(len(wavs))]
        mix_wav = sum(scaled_wav)
    return [mix_wav] + scaled_wav + skip_feats, ratios + skip_ratios


def make_wav_delay(
    wavs: list[torch.Tensor],
    delay_type: str,
    delay_config: dict,
):
    assert delay_type in ['fix_dur', 'flex_dur']

    if delay_type == 'fix_dur':
        delay_dur = delay_config.get('delay_dur', 3)
        if (random.random() < delay_config.get('delay_prob', 0.3)) and (len(wavs) > 1):
            delay_idx = random.randint(0, len(wavs) - 1)
            delay_len =  random.randint(16000, 16000 * delay_dur)

            ts = [0, delay_dur]
            te = [wavs[1 - delay_idx].shape[-1], wavs[delay_idx].shape[-1]]

            zero_padding = torch.zeros(1, delay_len)
            wavs[delay_idx] = torch.cat([zero_padding, wavs[delay_idx]], dim=1)
        else:
            ts = [0, 0]
            te = [wav.shape[-1] for wav in wavs]
        ovr_ratio = max(0, min(te) - max(ts)) / (max(te) - min(ts))
        return wavs, -1
    elif delay_type == 'flex_dur':
        # This overlap calculation supports two speakers only.
        attempted_ovr_ratio = random.random()

        # wavs[0] is the target; preserve source order.
        wav_lens = [_.shape[-1] for _ in wavs]
        min_len = min(wav_lens)
        max_len = max(wav_lens)
        max_avail_ovr_ratio = min_len / max_len

        ovr_ratio = min(attempted_ovr_ratio, max_avail_ovr_ratio)
        # Maximum achievable overlap ratio: min(source lengths) / max(source lengths).
        if attempted_ovr_ratio > max_avail_ovr_ratio:
            delay_idx = np.argmin([_.size(-1) for _ in wavs])
            delay_len = random.randint(0, max_len - min_len)
        else:
            delay_idx = random.randint(0, len(wavs) - 1)
            dice: float = random.random()
            if dice < delay_config.get('non_ovr_prob', 0.2):
                delay_len = wav_lens[delay_idx] + random.randint(16000, 64000)  # Insert a random 1–4 second gap for non-overlapping sources.
                ovr_ratio = 0
            else:
                # Overlap ratio = (other_length - delay) / (delay + delayed_length).
                delay_len = int((wav_lens[1 - delay_idx] - ovr_ratio * wav_lens[delay_idx]) / (1 + ovr_ratio))

        zero_padding = torch.zeros(1, delay_len)
        wavs[delay_idx] = torch.cat([zero_padding, wavs[delay_idx]], dim=1)
        return wavs, ovr_ratio


def padding_wav(wav, length, length_idx=1):
    if wav.size(length_idx) < length:
        wav_shape = wav.size()
        res_num = length - wav.size(length_idx)
        if length_idx == 0:
            d = wav_shape[1]
            padding_shape = (res_num, d)
        else:
            d = wav_shape[0]
            padding_shape = (d, res_num)
        padding = torch.zeros(padding_shape)
        wav = torch.cat([wav, padding], dim=length_idx)
    else:
        wav = wav[:,:length] if length_idx == 1 else wav[:length,:]
    return wav


def trim_wav(wav, segment):
    head, tail = segment
    if head == -1:
        # With head=-1, tail specifies the desired window length.
        head = 0
        wav_size = wav.size(1)
        res = wav_size - tail
        head = 0 if res <= 0 else int(random.uniform(0, res))
        tail = head + tail
    wav = wav[:,head: tail]
    return wav


def got_seg(v, segment):
    diff = float('inf')
    nearest_value, nearest_index = None, None
    for i, value in enumerate(segment):
        if abs(value - v) < diff:
            diff = abs(value - v)
            nearest_value = value
            nearest_index = i
    return nearest_value, nearest_index


def convert_mfa_to_align(mfa_obj):
    ali = []
    # Each phone maps to two flattened boundary indices: start=2*i, end=2*i+1.
    for (phn, pos) in mfa_obj:
        one_ali = [
            p for i, p in enumerate(phn)
            for _ in range(pos[i * 2], pos[i * 2 + 1] + 1)
        ]
        ali.extend(one_ali)
    return torch.tensor(ali)

# JSON source rows use json.loads; already loaded noise records pass through.


def make_corrupt_party(
        corrupt_list: list,
        corrupt_list_len: int,
        config: dict,
        corruption_material: list, detach_f=lambda x: x, prob=1.2,
    ):
    ratios = []
    n = config.get('num_corrupt')
    if config.get('random_num', False):
        assert(n > 1)
        n = RANDOM_FACTOR['int'](1, n+1)
    if config.get('prob', 1.2) < 1:
        dice = random.uniform(0,1)
        if dice > config.get('prob'):
            n = 0

    sampler = config.get('sampler')
    assert (sampler in RANDOM_FACTOR.keys())
    sampler_config = config.get('sampler_config')
    if len(corruption_material) == 0:
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
        c_idx = 0
    else:
        c_idx = len(corruption_material)

    for i in (range(n)):
        one_corrupt = corrupt_list[np.random.randint(0, corrupt_list_len)]
        one_corrupt = detach_f(one_corrupt)
        ratio = RANDOM_FACTOR[sampler](**sampler_config)
        ratio = ratio.tolist()[0] if isinstance(ratio, np.ndarray) else ratio
        ratios.append(ratio)
        if isinstance(one_corrupt, dict):
            corruption_material[c_idx+i+1] = one_corrupt
        elif isinstance(one_corrupt, str):
            corruption_material[c_idx+i+1] = {'sph': one_corrupt}
        else:
            raise NotImplementedError("corrupt list error")
    return ratios, corruption_material, n


def time_shifting(wav, frame_length=400, hop_length=160, shift_type='mid'):
    wav = wav.view(-1) # Mono waveform: [1, num_samples].
    num_frames = 1 + (len(wav) - frame_length) // hop_length
    energy = torch.zeros(num_frames, dtype=torch.float32)
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = wav[start:end]
        energy[i] = torch.sum(frame ** 2)

    idx_frame = torch.argmax(energy)
    idx_sample = idx_frame * hop_length + frame_length
    if shift_type == 'mid':
        dest_pos = int(wav.size(0) / 2)
    shift = int(dest_pos-idx_sample)
    wav = torch.roll(wav, shift)
    return wav.view(1, -1) # Restore the mono channel dimension: [1, num_samples].


def make_segment(segments, win_len, trim_type='raw', dither=False, sample_rate=16000):
    seg_head_tail = []
    idx_head_tail = []
    for seg in segments:
        if seg == -1:
            seg_head_tail.append([-1, int(win_len*sample_rate)])
            idx_head_tail.append([-1, -1])
            continue
        if trim_type == 'raw':
            seg_head = seg[0]
            if dither:
                seg_head = seg_head - random.uniform(0, 0.1)
                seg_head = seg_head if seg_head > 0 else 0
            seg_tail = seg_head + win_len
            seg_head_tail.append([int(seg_head*sample_rate), int(seg_tail*sample_rate)])
            idx_head_tail.append([0, len(seg)])
        elif trim_type == 'completetrim':
            leng_wav = seg[-1][-1]
            if leng_wav > win_len:
                seg_head_range = leng_wav - win_len
                _, seg_idx = got_seg(seg_head_range, [s[0] for s in seg])
                seg_head_idx = seg_idx if seg_idx <= 1 else np.random.randint(0, seg_idx-1)
                seg_head = seg[seg_head_idx][0]
                seg_tail = seg_head + win_len
                _, seg_tail_idx = got_seg(seg_tail, [s[1] for s in seg])
                seg_tail_idx = seg_tail_idx + 1 if seg_tail_idx < len(seg) - 1 else seg_tail_idx
                seg_tail = seg[seg_tail_idx][1]
            else:
                seg_head = 0
                seg_tail = seg[-1][-1]
                seg_head_idx = 0
                seg_tail_idx = len(seg)
            seg_head_tail.append([int(seg_head*sample_rate), int(seg_tail*sample_rate)])
            idx_head_tail.append([seg_head_idx, seg_tail_idx])
        else:
            raise NotImplementedError("Only support raw, xxx")
    return seg_head_tail, idx_head_tail


def detach_corruption(
        material: dict,
        segment_idx: list = None,
    ):
    keywords = []
    phn_labels = []
    labels = []
    for i, (_idx, info) in enumerate(material.items()):
        if 'word_keyword' in info:
            keywords.append(info['word_keyword'])
        if 'keyword' in info:
            keywords.append(info['keyword'])
        if 'label' in info:
            label = info['label']
            if segment_idx is not None:
                segment_head, segment_tail = segment_idx[i]
                label = label[segment_head: segment_tail]
            labels.append(label)
        if 'phn_label' in info:
            phn_label = info['phn_label']
            if segment_idx is not None:
                segment_head, segment_tail = segment_idx[i]
                phn_label = phn_label[segment_head: segment_tail]
            phn_labels.append(phn_label)
    return keywords, labels, phn_labels


def inject_special_token(
        keyword, keyword_length, label,
        positive=True, keyword_pos=None, special_token={}
    ):
    TEXT_SPEC_TOKEN.update(special_token)
    new_phn_label = copy.deepcopy(label)
    new_keyword = copy.deepcopy(keyword)

    if (not positive) and (TEXT_SPEC_TOKEN['punk'] is not None):
        new_phn_label = torch.tensor([TEXT_SPEC_TOKEN['punk'] for x in range(len(new_phn_label)//3)])

    if TEXT_SPEC_TOKEN['psok'] is not None:
        new_keyword.insert(0, [TEXT_SPEC_TOKEN['psok']])

    if TEXT_SPEC_TOKEN['peok'] is not None:
        new_keyword.insert(len(new_keyword), [TEXT_SPEC_TOKEN['peok']])

    if (TEXT_SPEC_TOKEN['with_trans']) and (positive):
        new_phn_label[keyword_pos: keyword_pos+keyword_length] = new_keyword
    return new_keyword, new_phn_label, keyword_pos


def snipe_edge(waveform, hop_length=160):
    num_samples = waveform.size(1)
    edges = num_samples % hop_length
    return waveform[:,0:num_samples-edges]


def process_raw(data: Generator[dict, None, None]):
    for sample in data:
        one_sample: dict = json.loads(sample['src'])

        epoch = sample['epoch']
        one_sample.update({'epoch': epoch})

        if one_sample.get('infer_wav', False):
            yield one_sample

        if 'self_corruption' in sample:
            self_corruption = sample['self_corruption']
            self_corruption = [json.loads(d) for d in self_corruption]
            one_sample.update({'self_corruption': self_corruption})
        if 'none_target_corruption' in sample:
            none_target_corruption = sample['none_target_corruption']
            one_sample.update({'none_target_corruption': none_target_corruption})
        if 'rirs' in sample:
            rirs_src = sample['rirs']
            one_sample.update({'rirs':rirs_src})
        if 'neg_candidate' in sample:
            neg_candidate = sample['neg_candidate']
            one_sample.update({'neg_candidate': neg_candidate})

        yield one_sample


def process_corruption(data: Generator[dict, None, None], config: dict, egs_format=False):
    for sample in data:
        if sample.get('infer_wav', False):
            yield sample

        corruption_material = {}
        corruption_ratios = []
        num_corrupt = n_scorrupt = n_ncorrupt = 0
        if config.get('self_corruption', False):
            assert 'self_corruption' in sample
            corrupt_list = sample['self_corruption']
            corrupt_list_len = len(corrupt_list)
            ratios, corruption_material, n_scorrupt = make_corrupt_party(
                corrupt_list, corrupt_list_len, config['self_corruption'], corruption_material,
            )
            corruption_ratios.extend(ratios)
            num_corrupt += n_scorrupt

        if config.get('none_target_corruption', False):
            assert 'none_target_corruption' in sample
            corrupt_list = sample['none_target_corruption']
            corrupt_list_len = len(corrupt_list)
            ratios, corruption_material, n_ncorrupt = make_corrupt_party(
                corrupt_list, corrupt_list_len, config['none_target_corruption'], corruption_material
            )
            corruption_ratios.extend(ratios)
            num_corrupt += n_ncorrupt

        sample.update({
            'corruption_ratios': corruption_ratios,
            'corruption_material': corruption_material,
            'n_scorrupt': n_scorrupt,
            'n_ncorrupt': n_ncorrupt,
            'num_corrupt': num_corrupt,
        })
        yield sample


def process_speech_feats(data: Generator[dict, None, None], config: dict, egs_format=False):
    for sample in data:
        input_data_type = config.get('data_type', 'raw')
        if (input_data_type != 'raw') and ('corruption_material' in sample):
            raise NotImplementedError("Only support corruption on waveforme")
        feats: list[str] = [sample['sph']]

        if 'corruption_material' in sample:
            corruption_material: dict[int, dict] = sample['corruption_material']
            corruption_feats = [corruption_material[x]['sph'] for x in corruption_material.keys()]
            corruption_segments = [
                corruption_material[x]['segment'] if 'segment' in corruption_material[x] else -1
                for x in corruption_material.keys()
            ]
            feats.extend(corruption_feats)

        if egs_format:
            input_data_type = 'empty'
        feats: list[tuple[torch.Tensor, int]] = [INPUT_DATA_LOADER[input_data_type](x) for x in feats]
        feats: list[torch.Tensor] = [INPUT_DATA_LOADER['rm_sr'](x) for x in feats] if input_data_type == 'raw' else feats

        if config.get('wav_augment', False):
            if 'rirs' in sample:
                rirs_src = sample['rirs']
            feats = [wav_augment(f, config.get('wav_augment'), rirs_src) for f in feats]

        if config.get('trim_config', False):
            if 'segment' in sample:
                segments = sample['segment']
            else:
                segments = [[0,0]]
            if 'corruption_segments' in locals().keys():
                segments += corruption_segments
            segments, segment_idx = make_segment(segments, **config.get("trim_config", {}))
            sample.update({"segment_idx": segment_idx})
            feats = [trim_wav(feats[i], segments[i]) for i in range(len(feats))]

        if 'corruption_material' in sample:
            mix_config: dict = config.get('mix_config', {})
            if (sample['n_scorrupt'] != 0) and (sample['n_ncorrupt'] != 0):
                n_ncorrupt = sample['n_ncorrupt']
                crpt_idx = random.randint(0, sample['n_scorrupt'])
                target_feats = feats[crpt_idx]
                noise_feats, noise_ratio = feats[-n_ncorrupt:], sample['corruption_ratios'][-n_ncorrupt:]
                tmp_ratio = [sample['corruption_ratios'][crpt_idx]] + noise_ratio
                mix_feats, _ = make_mix_wav([target_feats]+noise_feats, tmp_ratio, **mix_config)
                feats[crpt_idx] = mix_feats[0]

                skip_idx = n_ncorrupt # Noise is already mixed into a target; skip its standalone source below.
            else:
                skip_idx = None

            feats, sample['corruption_ratios'] = make_mix_wav(
                feats, sample['corruption_ratios'], skip_idx=skip_idx, **mix_config
            )
        if config.get('snipe_edge', False):
            hop_length = config['snipe_edge'].get('hop_length', 160)
            feats = [snipe_edge(f, hop_length) for f in feats]

        if config.get('return_raw', False):
            if 'corruption_material' in sample:
                raw_feats = copy.deepcopy(feats[1])
            else:
                raw_feats = copy.deepcopy(feats[0])
            sample.update({'raw_wav': raw_feats.squeeze(0)})

        if config.get('random_raw', False):
            sample_len1 = feats[1].size(1)
            dur = 16000 * 4
            sample_head = random.randint(0, sample_len1-dur-1) if sample_len1 > dur else 0
            raw_wav1 = copy.deepcopy(feats[1][:, sample_head:sample_head+dur])
            sample.update({'raw_wav1': raw_wav1.squeeze(0)})

            sample_len2 = feats[2].size(1)
            dur = 16000 * 4
            sample_head = random.randint(0, sample_len2-dur-1) if sample_len2 > dur else 0
            raw_wav2 = copy.deepcopy(feats[2][:, sample_head:sample_head+dur])
            sample.update({'raw_wav2': raw_wav2.squeeze(0)})

        feats_type: str = config.get('feats_type', 'fbank')
        feats_config: dict = config.get('feats_config', FBANK_DEFAULT_SETTING)
        feats: list[torch.Tensor] = [FEATS_EXTRACTOR[feats_type](f, **feats_config) for f in feats]

        if config.get('splice_config'):
            splice_config = config.get('splice_config')
            feats = [splice_feats(f, **splice_config) for f in feats]

        if config.get('subsample_rate'):
            feats = [f[::config.get('subsample_rate')] for f in feats]

        start_idx = 0
        if 'corruption_material' in sample:
            mix_feats = feats[start_idx]
            start_idx += 1
            sample.update({"mixspeech": mix_feats})
        else:
            sample.update({"speech": feats[0]})

        # Keep the scaled clean source waveforms for supervision.
        if sample.get("n_scorrupt", 0) > 0:
            clean_feats = feats[start_idx: start_idx+sample['n_scorrupt']+1]
            ratios = sample['corruption_ratios']

            ratios = ratios[0:sample['n_scorrupt']+1]
            clean_feats = torch.cat([x.unsqueeze(0) for x in clean_feats], dim=0)
            start_idx = sample['n_scorrupt'] + 1
            sample.update({"speech": clean_feats[0]})
            sample.update({"ratios": ratios})

        yield sample


def process_text_feats(data: Generator[dict, None, None], id2k=None):
    for sample in data:
        if ('label' in sample) and ('segment_idx' in sample):
            label = sample['label']
            segment_idx = sample['segment_idx']
            m_head, m_tail = segment_idx[0]
            label = label[m_head: m_tail]
            sample.update({'label': label})

        if 'corruption_material' in sample:
            # The c_ prefix denotes a corruption source.
            if 'segment_idx' in sample:
                c_segment_idx = sample['segment_idx'][1:]
            else:
                c_segment_idx = None
            c_keyword, c_label, c_phn_label = detach_corruption(sample['corruption_material'], c_segment_idx)
            if len(c_keyword) != 0:
                sample.update({"mix_keyword": sample['word_keyword'] + unfold_list(c_keyword)})
            if len(c_label) != 0:
                sample.update({"crpt_label": c_label})
            if len(c_phn_label) != 0:
                sample.update({"c_phn_label": c_phn_label})

        yield sample


def process_sampled_keyword_from_label(
        data: Generator[dict, None, None],
        positive_prob: float = 0.5,
        special_token: dict = {},
        sample_config: dict = {},
        neg_len=None, id2phone=None, lexicon=None,
):
    TEXT_SPEC_TOKEN.update(special_token)
    for sample in data:
        new_phn_label = copy.deepcopy(sample['phn_label'])

        kw, kw_pos, kw_length, pos, target = make_keyword(
            sample=sample, positive_prob=positive_prob,
            sample_config=sample_config, neg_len=neg_len,
        )
        kw, new_phn_label, kw_pos = inject_special_token(
            keyword=kw, keyword_length=kw_length, positive=pos, label=new_phn_label,
            keyword_pos=kw_pos, special_token=special_token
        )

        sample.update({'keyword': kw, 'phn_label': new_phn_label, 'target': target})
        yield sample


def make_keyword(sample: dict, positive_prob: float, sample_config: dict, neg_len=None):
    dice = random.uniform(0, 1)
    label = copy.deepcopy(sample['phn_label'])
    if dice > positive_prob:
        # A negative keyword must be absent from both target and interfering speech.
        if 'c_phn_label' in sample:
            crpt_label = sample['c_phn_label']
            mlabel = label + crpt_label
        else:
            mlabel = label
        kw = random_one_neg(
            neg_list=sample['neg_candidate'], neg_len=neg_len,
            pos_label=mlabel, spk_id=sample['key'].split('-')[0],
            sample_config=sample_config,
        )
        kw_pos = -1
        pos = False
        target = torch.tensor([0])
    else:
        kw_candidate = sample.get('kw_candidate', None)
        kw, kw_pos = sample_kw_from_label(
            label=label, kw_candidate=kw_candidate,
            sample_config=sample_config,
        )
        pos = True
        target = torch.tensor([1])
    return kw, kw_pos, len(kw), pos, target


def sample_kw_from_label(label, sample_config: dict, kw_candidate=None):
    keyword_length_range = sample_config.get('keyword_length_range', [2,6])
    continuous: bool = sample_config.get('continuous', True)

    if continuous:
        kw_len = random.randint(*keyword_length_range)
        kw_len = min(kw_len, len(label))

        kw_pos = random.randint(0, len(label) - kw_len)
        kw = label[kw_pos: kw_pos + kw_len]
    else:
        raise NotImplementedError
    return kw, kw_pos


def random_one_neg(neg_list: list, neg_len, pos_label: list, sample_config: dict, spk_id=None):
    neg = pos_label[0]

    flatten_label = unfold_list(pos_label)
    flatten_neg = unfold_list(neg)

    flatten_label = int2sym(flatten_label)
    flatten_neg = int2sym(flatten_neg)

    neg_spk = spk_id if spk_id is not None else -1

    while (" ".join(flatten_neg) in " ".join(flatten_label)) or (neg_spk == spk_id):
        one_neg_list: str = neg_list[random.randint(0, neg_len - 1)]
        one_neg_list: dict = json.loads(one_neg_list)
        if spk_id is not None:
            neg_spk = one_neg_list['key'].split('-')[0]
        one_neg_label = one_neg_list['phn_label']
        kw_candidate = one_neg_list.get('kw_candidate', None)
        neg, _ = sample_kw_from_label(
            label=one_neg_label, kw_candidate=kw_candidate,
            sample_config=sample_config,
        )
        flatten_neg = unfold_list(neg)
        flatten_neg = int2sym(flatten_neg)
    return neg


def process_fix_keyword(data: Generator[dict, None, None], special_token={}):
    for sample in data:
        if len(special_token) != 0:
            kw: list[int] = sample['keyword']
            phn_label: list[int] = sample['phn_label']
            target: torch.Tensor = torch.tensor([sample['target']])
            pos = (sample['target'] == 1)
            kw, label, _ = inject_special_token(
                keyword=kw, keyword_length=len(kw),
                label=phn_label, special_token=special_token,
                positive=pos,
            )
            sample.update({'keyword': kw, 'label': label, 'target': target})
        yield sample

# Keep labels grouped by word until keyword sampling; flatten afterwards.


def process_list_data(data: Generator[dict, None, None],):
    for sample in data:
        for key, value in sample.items():
            if key in NONE_TENSOR_KEY:
                continue
            if isinstance(value, list):
                value = unfold_list(value)
            if not isinstance(value, torch.Tensor):
                sample.update({key: torch.tensor(value)})
        yield sample

# Record sequence lengths before batch padding.


def make_length(data: Generator[dict, None, None]):
    for sample in data:
        length_info = {}
        for key, value in sample.items():
            if key in NONE_TENSOR_KEY:
                continue
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() == 0:
                continue
            new_key = "{}_len".format(key)
            length = value.size(0)
            length_info.update({new_key: torch.tensor(length)})
        sample.update(length_info)
        yield sample


def concat_tensor(data_list, seq_padding=False, padding_value=0):
    if seq_padding:
        tensor = pad_sequence(data_list, batch_first=True, padding_value=padding_value)
    else:
        tensor = torch.cat([x.unsqueeze(0) for x in data_list], dim=0)
    return tensor


def fetch_tensor(data: Generator[dict, None, None], fetch_key):
    for sample in data:
        if fetch_key[0] == 'key':
            sort_key = fetch_key[1]
        else:
            sort_key = fetch_key[0]
        index = torch.tensor([x[sort_key].size(0) for x in sample])
        index = torch.argsort(index, descending=True)
        return_feats = []
        for k in fetch_key:
            if (k == 'key') or (k == 'tag'):
                return_feats.append([sample[i][k] for i in index])
                continue
            if k in NONE_TENSOR_KEY:
                continue
            if sample[0][k].dim() != 0:
                seq_padding = True
            else:
                seq_padding = False
            if k in CTC_KEY:
                padding_value = -1
            else:
                padding_value = 0
            return_feats.append(
                concat_tensor(
                    [sample[i][k] for i in index], seq_padding=seq_padding, padding_value=padding_value
                )
            )
        yield tuple(return_feats)


def process_speaker_label(data: Generator[dict, None, None], spk2cls: str):
    spk2cls = json.load(open(spk2cls))
    for sample in data:
        spk_key = sample['key'].split('-')[0]
        spk_target = torch.tensor(spk2cls[spk_key])
        sample.update({'spk_label': spk_target})
        yield sample


def make_batch(data: Generator[dict, None, None], batch_size=256):
    buf = []

    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf
