# Experiments

This folder contains the reproducible experiment scripts and generated outputs for the project.

| # | Script | Output |
|---:|---|---|
| 01 | [experiment1.py](experiment1.py) | `result.png` |
| 02 | [experiment2.py](experiment2.py) | console metrics |
| 03 | [experiment3.py](experiment3.py) | `feature_importance.png` |
| 04 | [experiment4_model_comparison.py](experiment4_model_comparison.py) | console model comparison |
| 05 | [experiment5_feature_removal.py](experiment5_feature_removal.py) | console sensitivity metrics |
| 06 | [experiment6_noise_curve.py](experiment6_noise_curve.py) | `noise_curve.png` |
| 07 | [experiment7_threshold_analysis.py](experiment7_threshold_analysis.py) | `threshold_analysis.png` |
| 08 | [experiment8_comparison.py](experiment8_comparison.py) | `failure_comparison.png` |
| 09 | [experiment9_multi_run_analysis.py](experiment9_multi_run_analysis.py) | `model_statistics.csv` |
| 10 | [experiment10_failure_matrix.py](experiment10_failure_matrix.py) | `failure_matrix.csv`, `failure_matrix.png` |
| 11 | [experiment11_confidence_collapse.py](experiment11_confidence_collapse.py) | `confidence_collapse.csv`, `confidence_collapse.png` |
| 12 | [experiment12_refusal_system.py](experiment12_refusal_system.py) | `refusal_statistics.csv`, `accuracy_vs_coverage.png` |
| 13 | [experiment13_calibration.py](experiment13_calibration.py) | `calibration_metrics.csv`, `calibration_bins.csv`, `calibration_curve.png`, `reliability_diagram.png` |
| 14 | [experiment14_reliability_score.py](experiment14_reliability_score.py) | `reliability_scores.csv`, `reliability_run_metrics.csv`, `reliability_scores.png` |
| 15 | [experiment15_distribution_shift.py](experiment15_distribution_shift.py) | `shift_statistics.csv`, `distribution_shift.png` |
| 16 | [experiment16_reliability_index.py](experiment16_reliability_index.py) | `reliability_index.csv`, `reliability_index.png` |
| 17 | [experiment17_model_ranking.py](experiment17_model_ranking.py) | `model_ranking.csv`, `model_ranking.png` |

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
