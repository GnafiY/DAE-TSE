import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import os
import yaml

import torchaudio.compliance.kaldi as kaldi

from kce.model.AEDKWSASRPhone import AEDKWSASRPhone
import kce.model.NetModules as NM


def get_asr_encoder(conf: dict):
    conf = dict(conf)
    model_name = conf.pop('model_name')
    model_dict = {
        'TCASREncoder': TCASREncoder,
    }
    return model_dict[model_name](**conf)


class TCASREncoder(nn.Module):
    def __init__(self, checkpoint_path: str, fnn: dict, multi_task: bool = False, spksInTrain: int = 251,
                 backbone_freeze: bool = True, eps: float = 1e-5):
        super().__init__()

        # wav -> fbank
        self.frontend = kaldi.fbank
        self.frontend_conf = self.load_frontend(checkpoint_path)
        self.frame_shift_s = float(self.frontend_conf.get("frame_shift", 10)) / 1000.0

        # load & freeze TC-ASR model
        self.tcasr_encoder = self.load_checkpoint(checkpoint_path)
        if backbone_freeze:
            for param in self.tcasr_encoder.parameters():
                param.requires_grad = False

        # adapter
        num_layer = fnn['num_layer']
        if num_layer == 1:
            self.fnn = nn.Sequential(*[
                nn.Linear(fnn['input_dim'], fnn['output_dim']),
            ])
        else:
            self.fnn = []
            for _ in range(num_layer - 1):
                self.fnn.extend([
                    nn.Linear(fnn['input_dim'], fnn['input_dim']),
                ])
            self.fnn.extend([
                nn.Linear(fnn['input_dim'], fnn['output_dim']),
            ])
            self.fnn = nn.Sequential(*self.fnn)

        # mutli_task: predict speaker
        self.multi_task = multi_task
        if multi_task:
            self.spk_proj = nn.Linear(fnn['output_dim'], spksInTrain)

    def load_frontend(self, checkpoint_path) -> dict:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        exp_dir = os.path.dirname(checkpoint_path)

        config_file = '{}/data.yaml'.format(exp_dir)
        data_config = yaml.load(open(config_file), Loader=yaml.FullLoader)

        return data_config['speech_config']['feats_config']

    def load_checkpoint(self, checkpoint_path) -> nn.Module:
        checkpoint_path = os.path.expanduser(checkpoint_path)
        exp_dir = os.path.dirname(checkpoint_path)

        config_file = '{}/model.yaml'.format(exp_dir)
        model_config = yaml.load(open(config_file), Loader=yaml.FullLoader)

        model = AEDKWSASRPhone(**model_config)
        state_dict = torch.load(
            checkpoint_path,
            weights_only=True,
            map_location='cpu',
        )['model']
        model.load_state_dict(state_dict)

        return model

    def forward(self, wav_input: torch.Tensor, text_input: list,
                return_attention: bool = False, attn_layer: int = -1):
        fbank = [self.frontend(_.unsqueeze(0), **self.frontend_conf) for _ in wav_input]
        fbank_len = torch.tensor([_.shape[0] for _ in fbank])
        fbank = pad_sequence(fbank, batch_first=True)
        fbank_len = fbank_len.to(fbank.device)

        if return_attention:
            sph_emb, attn_pack = self.forward_tcasr(
                (fbank, fbank_len), text_input,
                return_attention=True, attn_layer=attn_layer)
        else:
            sph_emb = self.forward_tcasr((fbank, fbank_len), text_input)
            attn_pack = None

        pooled_sph_emb = self.forward_pooling(sph_emb)  # B, D
        speaker_emb = self.fnn(pooled_sph_emb)

        if self.multi_task:
            speaker_pred = self.spk_proj(speaker_emb)
        else:
            speaker_pred = None

        if return_attention:
            return speaker_emb, speaker_pred, attn_pack
        return speaker_emb, speaker_pred

    @torch.no_grad()
    def forward_tcasr(self, wav_input, text_input, return_attention: bool = False,
                      attn_layer: int = -1):
        sph_input, sph_len = wav_input
        kw_label, kw_len = text_input
        b, t, d = sph_input.size()

        sph_len = NM.BaseConv.compute_dim_reduction(sph_len, 3, 2, 0, 1)
        sph_len = NM.BaseConv.compute_dim_reduction(sph_len, 3, 2, 0, 1)
        sph_mask = ~NM.make_mask(sph_len).unsqueeze(1)
        kw_mask = ~NM.make_mask(kw_len).unsqueeze(1)
        cross_mask = ~NM.combine_mask(sph_mask.squeeze(1), kw_mask.squeeze(1), 1)

        # embedding
        sph_emb = self.tcasr_encoder.au_conv(sph_input.unsqueeze(1))
        b, c, t, d = sph_emb.size()
        sph_emb = self.tcasr_encoder.au_conv_trans(sph_emb.transpose(1, 2).contiguous().view(b, t, c * d))
        sph_emb = self.tcasr_encoder.speech_input_projection(sph_emb)
        kw_emb = self.tcasr_encoder.phn_emb(kw_label.to(torch.long))
        kw_emb = self.tcasr_encoder.keyword_input_projection(kw_emb)

        # add position embedding
        sph_emb = self.tcasr_encoder.speech_pe_module(sph_emb)
        kw_emb = self.tcasr_encoder.keyword_pe_module(kw_emb)

        kw_emb = self.tcasr_encoder.forward_transformer(
            self.tcasr_encoder.keyword_transformer,
            kw_emb,
            attention_mask=kw_mask,
        )
        # analyse=True is required: speaker emb uses sph_emb_list[-2]
        sph_emb, (sph_att_scores, sph_emb_list) = self.tcasr_encoder.forward_transformer(
            self.tcasr_encoder.speech_transformer,
            sph_emb,
            attention_mask=sph_mask,
            cross_hidden_states=(
                kw_emb, kw_emb, cross_mask
            ),
            analyse=True,
        )

        '''
            sph_att_scores: dict, with range(batch_size) as key,
                list[layer] of [n_head, T_speech, T_kw]
            sph_emb_list:   dict, with range(batch_size) as key
        '''
        sph_emb_list = sph_emb_list[0]  # omit repeated sph_emb

        if self.tcasr_encoder.use_sv:
            sph_emb = self.tcasr_encoder.sv_ce_crit.forward_pooling(sph_emb_list, sph_len)
        else:
            sph_emb = sph_emb_list[-2]

        if not return_attention:
            return sph_emb

        n_layers = len(sph_att_scores[0])
        layer_idx = attn_layer if attn_layer >= 0 else n_layers + attn_layer
        att_list = []
        for bidx in range(b):
            t_len = int(sph_len[bidx].item())
            k_len = int(kw_len[bidx].item())
            att = sph_att_scores[bidx][layer_idx][:, :t_len, :k_len]
            att = att.mean(dim=0).detach().cpu()
            att_list.append(att)

        attn_pack = {
            "attention": att_list,  # list of [T, K]
            "attention_map": att_list,  # alias
            "speech_len": sph_len.detach().cpu(),
            "kw_len": kw_len.detach().cpu(),
            "kw_label": kw_label.detach().cpu(),
            "frame_shift_s": self.frame_shift_s,
            "layer_idx": layer_idx,
        }
        return sph_emb, attn_pack

    def forward_pooling(self, sph_emb: torch.Tensor):
        if self.tcasr_encoder.use_sv:     # B x D
            return sph_emb
        pooled_sph_emb = sph_emb.mean(dim=1)    # B, T_max, D
        # select emb, not all
        return pooled_sph_emb   # B, D
