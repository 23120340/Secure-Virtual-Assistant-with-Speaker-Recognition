# ECAPA-TDNN Training Dataset Comparison

Generated from the training artifacts under `training/results/`.

## Scope

- Model: ECAPA-TDNN + AAM-Softmax.
- Feature: 80-dim log Mel filterbank with CMN.
- Datasets: VoxCeleb Indian subset and VIVOS.
- MUSAN/RIR are excluded from the main evidence because the noise/RIR datasets are not available in this experiment.

## Main Results

| Dataset | Speakers | Utterances | Train / Val / Test | Epochs | Best Val Acc | SID Top-1 | SID Top-5 | SV EER | minDCF | Threshold @ EER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VoxCeleb Indian | 24 | 4857 | 3389 / 717 / 751 | 15 | 93.44% | 91.34% | 98.67% | 2.90% | 0.1486 | 0.3884 |
| VIVOS | 65 | 12419 | 8686 / 1850 / 1883 | 15 | 99.19% | 99.31% | 100.00% | 0.96% | 0.0673 | 0.4062 |

## Trial Counts

| Dataset | Trials | Positive | Negative |
|---|---:|---:|---:|
| VoxCeleb Indian | 552 | 276 | 276 |
| VIVOS | 4160 | 2080 | 2080 |

## Discussion Template

- Compare whether VIVOS improves Vietnamese-domain speaker recognition or only closed-set SID.
- Explain that EER/minDCF are internal metrics from generated trial pairs, not official VoxCeleb1-O or NIST SRE benchmarks.
- If VIVOS is better, attribute likely causes to language/domain match and possibly smaller/easier speaker set.
- If VoxCeleb is better, attribute likely causes to greater speaker/audio variation despite domain mismatch.
- State that MUSAN/RIR augmentation was implemented in code but excluded from reported evidence due to unavailable auxiliary datasets.

## Artifact Folders

- VoxCeleb Indian: `training\results\voxceleb_indian`
- VIVOS: `training\results\vivos`
