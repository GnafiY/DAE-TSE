# Original LibriSpeech KCE Recipe

See the repository [training guide](../../../README.md#stage-a-train-kce-on-librispeech)
for the original train-860 / validation-100 protocol, environment, loss and
whole-utterance launch commands. `local/` contains the migrated preparation
scripts; `conf/train.yaml` records the original active training settings.

Stage 0 resumes each subset independently at WAV-list, duration, text-label and
JSONL preparation steps. Successful outputs have adjacent `.done` files; a
failed step is rerun on the next launch. Combined train/dev lists and speaker
labels use the same convention. Rerun the original stage-0 command to continue.
To rebuild a completed step deliberately, remove its `.done` file and those of
any downstream outputs that need rebuilding. No checksums are used.
