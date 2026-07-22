# Experiments

This folder contains the reproducible experiment scripts for the project.

Generated CSV files are saved to `../results/`. Generated figures are saved to `../figures/`, with paper-ready copies in `../paper/figures/`.

| # | Script | Output |
|---:|---|---|
| 01 | [experiment1.py](experiment1.py) | `../figures/result.png` |
| 02 | [experiment2.py](experiment2.py) | console metrics |
| 03 | [experiment3.py](experiment3.py) | `../figures/feature_importance.png` |
| 04 | [experiment4_model_comparison.py](experiment4_model_comparison.py) | console model comparison |
| 05 | [experiment5_feature_removal.py](experiment5_feature_removal.py) | console sensitivity metrics |
| 06 | [experiment6_noise_curve.py](experiment6_noise_curve.py) | `../figures/noise_curve.png` |
| 07 | [experiment7_threshold_analysis.py](experiment7_threshold_analysis.py) | `../figures/threshold_analysis.png` |
| 08 | [experiment8_comparison.py](experiment8_comparison.py) | `../figures/failure_comparison.png` |
| 09 | [experiment9_multi_run_analysis.py](experiment9_multi_run_analysis.py) | `../results/model_statistics.csv` |
| 10 | [experiment10_failure_matrix.py](experiment10_failure_matrix.py) | `../results/failure_matrix.csv`, `../figures/failure_matrix.png` |
| 11 | [experiment11_confidence_collapse.py](experiment11_confidence_collapse.py) | `../results/confidence_collapse.csv`, `../figures/confidence_collapse.png` |
| 12 | [experiment12_refusal_system.py](experiment12_refusal_system.py) | `../results/refusal_statistics.csv`, `../figures/accuracy_vs_coverage.png` |
| 13 | [experiment13_calibration.py](experiment13_calibration.py) | `../results/calibration_metrics.csv`, `../results/calibration_bins.csv`, `../figures/calibration_curve.png`, `../figures/reliability_diagram.png` |
| 14 | [experiment14_reliability_score.py](experiment14_reliability_score.py) | `../results/reliability_scores.csv`, `../results/reliability_run_metrics.csv`, `../figures/reliability_scores.png` |
| 15 | [experiment15_distribution_shift.py](experiment15_distribution_shift.py) | `../results/shift_statistics.csv`, `../figures/distribution_shift.png` |
| 16 | [experiment16_reliability_index.py](experiment16_reliability_index.py) | `../results/reliability_index.csv`, `../figures/reliability_index.png` |
| 17 | [experiment17_model_ranking.py](experiment17_model_ranking.py) | `../results/model_ranking.csv`, `../figures/model_ranking.png` |

Run the newest experiments:

```bash
python experiments/experiment9_multi_run_analysis.py
python experiments/experiment10_failure_matrix.py
python experiments/experiment11_confidence_collapse.py
python experiments/experiment12_refusal_system.py
python experiments/experiment13_calibration.py
python experiments/experiment14_reliability_score.py
python experiments/experiment15_distribution_shift.py
python experiments/experiment16_reliability_index.py
python experiments/experiment17_model_ranking.py
```
