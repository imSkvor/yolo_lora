import json
import pandas as pd
from pathlib import Path


def build_experiment_table(experiments_dir: Path) -> pd.DataFrame:
    """Creates a comparison table of all experiments for README"""
    rows: list = []
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
    
        metrics_file = exp_dir / "metrics.json"
        config_file = exp_dir / "config.yaml" # fix this with the current structure + GPU monitor

        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)

            row = {
                "Experiment": exp_dir.name,
                "Model": exp_dir.name.split('_')[0],
                "mAP50": metrics.get("metrics/mAP50", "N/A"),
                "mAP50-95": metrics.get("metrics/mAP50-95", "N/A"),
                "Params (M)": metrics.get("params/trainable", "N/A"),
                "VRAM (GB)": metrics.get("hardware/vram_usage", "N/A"),
                "Training Time": metrics.get("time/total_hours", "N/A")
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_markdown("experiments_comparison.md", index = False)
    return df
