# KSHPV derived data and analysis code

This repository contains de-identified derived data and analysis code for the manuscript:

**Phase-aware kinematic measurement of standing long jump based on human posture vision**

## Contents

- `data/derived_phase_indicators_deidentified.csv`: de-identified pose-derived phase indicators and participant-level covariates used for statistical analysis. Raw video file names, paths, and subject identifiers have been removed.
- `data/recommended_feature_set.json`: feature set used by the primary random-forest analysis.
- `data/table_data/`: CSV files underlying the manuscript tables.
- `data/figure_data/`: CSV files underlying the manuscript figures.
- `data/data_dictionary.csv`: column descriptions for the released derived dataset.
- `scripts/`: Python scripts used for model validation, hypothesis testing, robustness analysis, sensitivity analysis, and table/figure generation.

## Data availability note

The raw side-view video recordings are not included because they contain identifiable human-subject information and are subject to privacy and ethical restrictions. This release provides de-identified derived variables and table/figure data sufficient to inspect and reproduce the reported statistical analyses.

## Basic setup

```bash
pip install -r requirements.txt
```

## Example reproduction commands

Run the primary grouped validation and permutation feature-importance analysis:

```bash
python scripts/run_paper_main_analysis.py \
  --features-csv data/derived_phase_indicators_deidentified.csv \
  --feature-set-json data/recommended_feature_set.json \
  --target-col distance_cm \
  --group-col participant_id \
  --cv-splits 5 \
  --seed 42 \
  --rf-n-estimators 300 \
  --perm-repeats 30 \
  --out-dir outputs/paper_main
```

Run hypothesis tests:

```bash
python scripts/run_hypothesis_tests.py \
  --features-csv data/derived_phase_indicators_deidentified.csv \
  --feature-set-json data/recommended_feature_set.json \
  --target-col distance_cm \
  --group-col participant_id \
  --out-dir outputs/hypothesis
```

Run subgroup robustness analysis:

```bash
python scripts/run_robustness_subgroups.py \
  --features-csv data/derived_phase_indicators_deidentified.csv \
  --feature-set-json data/recommended_feature_set.json \
  --target-col distance_cm \
  --group-col participant_id \
  --out-dir outputs/robustness
```
