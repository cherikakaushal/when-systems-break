import numpy as np
import pandas as pd


def estimate_failure_risk(confidence, noise_level, missing_ratio, operation_count):
    risk = (1 - confidence) * 0.55
    risk += noise_level * 0.25
    risk += missing_ratio * 0.15
    risk += min(operation_count * 0.05, 0.20)
    return min(max(float(risk), 0), 1)


def repository_stats(catalog, tables):
    figure_count = sum(len(item["figures"]) for item in catalog)
    csv_count = sum(len(item["csv"]) for item in catalog)
    return {
        "Experiments": len(catalog),
        "CSV Outputs": csv_count,
        "Figures": figure_count,
        "Result Tables": len(tables),
    }


def latest_findings(tables):
    ranking = tables.get("model_ranking", pd.DataFrame())
    calibration = tables.get("calibration_metrics", pd.DataFrame())
    shift = tables.get("shift_statistics", pd.DataFrame())
    findings = []
    if not ranking.empty:
        leader = ranking.iloc[0]
        findings.append(f"{leader['Model']} is the top-ranked model with a Reliability Index of {leader['Reliability Index']:.2f}.")
    if not calibration.empty:
        best = calibration.sort_values("Expected Calibration Error").iloc[0]
        findings.append(f"{best['Model']} has the lowest observed calibration error under {best['Condition'].lower()} inputs.")
    if not shift.empty:
        endpoint = shift[shift["Mean Shift"] == shift["Mean Shift"].max()]
        strongest = endpoint.sort_values("Accuracy Mean", ascending=False).iloc[0]
        findings.append(f"{strongest['Model']} retains the highest endpoint accuracy under distribution shift.")
    return findings


def confidence_meter_label(confidence):
    if confidence >= 0.9:
        return "High confidence"
    if confidence >= 0.7:
        return "Moderate confidence"
    return "Low confidence"


def reliability_label(score):
    if np.isnan(score):
        return "Not available"
    if score >= 95:
        return "High reliability"
    if score >= 90:
        return "Moderate reliability"
    return "Needs review"

