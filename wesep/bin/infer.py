from __future__ import print_function

import os
import time

import fire
import numpy
import soundfile
import torch
import yaml
from torch.utils.data import DataLoader
from datetime import datetime
import json

from wesep.dataset.dataset import Dataset, tse_collate_fn
from wesep.models import get_model
from wesep.utils.checkpoint import load_pretrained_model
from wesep.utils.score import cal_SISNRi
from wesep.utils.utils import (
    generate_enahnced_scp,
    get_logger,
    parse_config_or_kwargs,
    set_seed,
)
from wesep.utils.executor import process_additional


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise ValueError(f'Cannot convert {v}({type(v)}) into bool.')


def infer(config="confs/conf.yaml", **kwargs):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_cnt = 0
    accept_cnt = 0

    configs = parse_config_or_kwargs(config, **kwargs)
    sign_save_wav = configs.get(
        "save_wav", True)  # Control if save the extracted speech as .wav
    sign_save_wav = str2bool(sign_save_wav)

    rank = 0
    set_seed(configs["seed"] + rank)
    gpu = configs["gpus"]
    device = (torch.device("cuda:{}".format(gpu))
              if gpu >= 0 else torch.device("cpu"))

    sample_rate = configs.get("fs", None)
    if sample_rate is None or sample_rate == "16k":
        sample_rate = 16000
    else:
        sample_rate = 8000

    if 'spk_model_init' in configs['model_args']['tse_model']:
        configs['model_args']['tse_model']['spk_model_init'] = False
    model = get_model(
        configs["model"]["tse_model"])(**configs["model_args"]["tse_model"])
    model_path = os.path.join(configs["checkpoint"])
    load_pretrained_model(model, model_path)

    infer_tag = configs.get('infer_tag', datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
    configs["exp_dir"] = os.path.join(configs["exp_dir"], 'inference', infer_tag)
    os.makedirs(os.path.join(configs["exp_dir"]), exist_ok=True)

    jsonl_log = os.path.join(configs["exp_dir"], f"jsonl_split/infer.{gpu}.jsonl")
    os.makedirs(os.path.dirname(jsonl_log), exist_ok=True)

    logger = get_logger(configs["exp_dir"], "infer.log")
    logger.info("Load checkpoint from {}".format(model_path))
    save_audio_dir = os.path.join(configs["exp_dir"], "audio")

    # if sign_save_wav:
    #     if not os.path.exists(save_audio_dir):
    #         try:
    #             os.makedirs(save_audio_dir)
    #             print(f"Directory {save_audio_dir} created successfully.")
    #         except OSError as e:
    #             print(f"Error creating directory {save_audio_dir}: {e}")
    #     else:
    #         print(f"Directory {save_audio_dir} already exists.")
    # else:
    #     print("Do NOT save the results in wav.")

    if sign_save_wav:
        os.makedirs(save_audio_dir, exist_ok=True)
        print(f"Directory {save_audio_dir} created successfully.")
    else:
        print("Do NOT save the results in wav.")


    model = model.to(device)
    model.eval()

    assert configs.get("test_text_enroll"), (
        "DAE-TSE inference requires --test_text_enroll (keyword cue JSONL).")
    additional_conf = {
        "data_type": "TC-ASR",
        "kw_sampler": "fix",
        "data_list": configs["test_text_enroll"],
    }
    configs.setdefault("dataset_args", {}).setdefault("additional", {})
    configs["dataset_args"]["additional"]["test"] = additional_conf

    test_dataset = Dataset(
        configs.get("data_type", "raw"),
        configs["test_data"],
        configs.get("dataset_args", {}),
        state="test",
        whole_utt=configs.get("whole_utt", True),
        additional_conf=additional_conf,
    )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 collate_fn=tse_collate_fn)
    test_iter = len(open(configs["test_data"], 'r').readlines())
    logger.info("test number: {}".format(test_iter))

    # save config.yaml
    saved_config_path = os.path.join(configs["exp_dir"], "inference.yaml")
    with open(saved_config_path, "w") as fout:
        data = yaml.dump(configs)
        fout.write(data)

    jsonl_log_lines = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            features = batch["wav_mix"]
            targets = batch["wav_targets"]
            enroll = batch["spk_embeds"]
            spk = batch["spk"]
            key = batch["key"]

            features = features.float().to(device)  # (B,T,F)
            targets = targets.float().to(device)
            enroll = enroll.float().to(device)

            additional = process_additional(batch, device)

            outputs = model(features, enroll, additional)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if torch.min(outputs.max(dim=1).values) > 0:
                outputs = ((outputs /
                            abs(outputs).max(dim=1, keepdim=True)[0] *
                            0.9).cpu().numpy())
            else:
                outputs = outputs.cpu().numpy()

            if sign_save_wav:
                file1 = os.path.join(
                    save_audio_dir,
                    f"Utt{total_cnt + 1}-{key[0]}-T{spk[0]}.wav",
                )
                soundfile.write(file1, outputs[0], sample_rate)
                file2 = os.path.join(
                    save_audio_dir,
                    f"Utt{total_cnt + 1}-{key[1]}-T{spk[1]}.wav",
                )
                soundfile.write(file2, outputs[1], sample_rate)

            refs = targets.cpu().numpy()
            ests = outputs
            mixs = features.cpu().numpy()

            ## 2 speakers
            for spk_idx in range(refs.shape[0]):
                if ests[spk_idx].size != refs[spk_idx].size:
                    end = min(ests[spk_idx].size, refs[spk_idx].size, mixs[spk_idx].size)
                    est = ests[spk_idx][:end]
                    ref = refs[spk_idx][:end]
                    mix = mixs[spk_idx][:end]
                    sisnr, delta = cal_SISNRi(est, ref, mix)
                else:
                    sisnr, delta = cal_SISNRi(ests[spk_idx], refs[spk_idx], mixs[spk_idx])

                # detect target confusion problem
                sisnr_interf, delta_interf = -numpy.inf, -numpy.inf
                for interf_idx in range(refs.shape[0]):
                    if interf_idx == spk_idx:
                        continue

                    # sisnr_interf, delta_interf = cal_SISNRi(mix[spk_idx], ref[spk_idx], mix[spk_idx])
                    sisnr_interf, delta_interf = cal_SISNRi(ests[spk_idx], refs[interf_idx], mixs[interf_idx])
                    sisnr_interf = max(sisnr_interf, sisnr_interf)
                    delta_interf = max(delta_interf, delta_interf)

                logger.info(
                    "Num={} | Utt={} | Target speaker={} | SI-SNR={:.2f} | SI-SNRi={:.2f}"
                    .format(total_cnt + 1, key[spk_idx], spk[spk_idx], sisnr, delta))
                
                jsonl_line = {
                    "rank": gpu,
                    "num": total_cnt,
                    "utt": key[spk_idx],
                    "target_speaker": spk[spk_idx],
                    "SI-SNR": float(f"{sisnr:.2f}"),
                    "SI-SNR_interf": float(f"{sisnr_interf:.2f}"),
                    "SI-SNRi": float(f"{delta:.2f}"),
                    "SI-SNRi_interf": float(f"{delta_interf:.2f}"),
                }
                jsonl_log_lines.append(json.dumps(jsonl_line) + "\n")
                
                total_SISNR += sisnr
                total_SISNRi += delta
                total_cnt += 1
                if delta > 1:
                    accept_cnt += 1

            # if (i + 1) == test_iter:
            #     break
        end = time.time()
    # generate the scp file of the enhanced speech for scoring
    if sign_save_wav:
        generate_enahnced_scp(os.path.abspath(save_audio_dir), extension="wav")

    logger.info("Time Elapsed: {:.1f}s".format(end - start))
    logger.info("Average SI-SNR: {:.2f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.2f}".format(total_SISNRi / total_cnt))
    logger.info(
        "Acceptance rate of Utterances with SI-SDRi > 1 dB: {:.2f}".format(
            accept_cnt / total_cnt * 100))

    with open(jsonl_log, "w") as fout:
        fout.writelines(jsonl_log_lines)


if __name__ == "__main__":
    fire.Fire(infer)
