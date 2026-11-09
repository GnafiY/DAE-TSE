#!/bin/bash
# Prepare LibriMix test wav.scp (WeSep-style meta files).
# Enrollment maps are NOT required for DAE-TSE inference.

stage=1
stop_stage=1

mix_data_path=/path/to/Libri2Mix/wav16k/min
data=data
noise_type=clean

. "$(dirname -- "${BASH_SOURCE[0]}")/parse_options.sh" || exit 1

data=$(realpath ${data})

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare the meta files for the test set"

  for dataset in test; do
    echo "Preparing files for" $dataset

    dataset_path=$mix_data_path/$dataset/mix_${noise_type}
    mkdir -p "${data}"/$noise_type/${dataset}
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print $NF}' |
      awk -v path="${dataset_path}" '{print $1 , path "/" $1 , path "/../s1/" $1 , path "/../s2/" $1}' |
      sed 's#.wav##' | sort -k1,1 >"${data}"/$noise_type/${dataset}/wav.scp
    awk '{print $1}' "${data}"/$noise_type/${dataset}/wav.scp |
      awk -F[_-] '{print $0, $1,$4}' >"${data}"/$noise_type/${dataset}/utt2spk

    # Reference sources (optional bookkeeping; SI-SNR refs come from wav.scp columns)
    dataset_path=$mix_data_path/$dataset/s1
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print "s1/" $NF, $0}' | sort -k1,1 >"${data}"/$noise_type/${dataset}/single.wav.scp
    awk '{print $1}' "${data}"/$noise_type/${dataset}/single.wav.scp | grep 's1' |
      awk -F[-_/] '{print $0, $2}' >"${data}"/$noise_type/${dataset}/single.utt2spk

    dataset_path=$mix_data_path/$dataset/s2
    find ${dataset_path}/ -type f -name "*.wav" | awk -F/ '{print "s2/" $NF, $0}' | sort -k1,1 >>"${data}"/$noise_type/${dataset}/single.wav.scp
    awk '{print $1}' "${data}"/$noise_type/${dataset}/single.wav.scp | grep 's2' |
      awk -F[-_/] '{print $0, $5}' >>"${data}"/$noise_type/${dataset}/single.utt2spk
  done
fi
