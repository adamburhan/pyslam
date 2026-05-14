import json
import pandas as pd
from pathlib import Path
import argparse

ENVS = [
    "abandonedfactory", "abandonedfactory_night", "amusement", "carwelding",
    "endofworld", "gascola", "hospital", "japanesealley", "neighborhood",
    "ocean", "office", "office2", "oldtown", "seasidetown",
    "seasonsforest", "seasonsforest_winter", "soulcity", "westerndesert"
]

PRESETS = [
    "baseline",
    "depth_gt_1", "depth_gt_2", "depth_gt_3",
    "depth_est_1", "depth_est_2", "depth_est_3"
]

PRESET_META = {
    "baseline":    {"depth_source": "none", "lambda_val": float("nan")},
    "depth_gt_1":  {"depth_source": "gt",   "lambda_val": 0.01},
    "depth_gt_2":  {"depth_source": "gt",   "lambda_val": 0.1},
    "depth_gt_3":  {"depth_source": "gt",   "lambda_val": 1.0},
    "depth_est_1": {"depth_source": "est",  "lambda_val": 0.01},
    "depth_est_2": {"depth_source": "est",  "lambda_val": 0.1},
    "depth_est_3": {"depth_source": "est",  "lambda_val": 1.0},
}

N_SEEDS = 10


def parse_other_metrics(path):
    """Parse other_metrics_info.txt → dict."""
    result = {}
    try:
        for line in path.read_text().strip().splitlines():
            key, val = line.split(":")
            key = key.strip()
            val = val.strip()
            result[key] = int(val) if key != "percent_lost" else float(val)
        return result
    except Exception:
        return None


def parse_stats_final(path):
    """Parse plot/stats_final.json → dict with ate_ prefix."""
    try:
        raw = json.loads(path.read_text())
        return {f"ate_{k}": v for k, v in raw.items()}
    except Exception:
        return None


def collect_all(data_root):
    rows = []
    for env in ENVS:
        for preset in PRESETS:
            preset_dir = data_root / env / "pyslam_outputs" / preset
            if not preset_dir.is_dir():
                continue
            sequences = sorted([d.name for d in preset_dir.iterdir() if d.is_dir()])
            for seq in sequences:
                for seed in range(N_SEEDS):
                    iter_dir = preset_dir / seq / f"iteration_{seed}"
                    row = {
                        "env": env,
                        "preset": preset,
                        "sequence": seq,
                        "seed": seed,
                        **PRESET_META[preset],
                        "iteration_path": str(iter_dir),
                    }

                    stats_path = iter_dir / "plot" / "stats_final.json"
                    metrics_path = iter_dir / "other_metrics_info.txt"

                    stats = parse_stats_final(stats_path) if stats_path.exists() else None
                    metrics = parse_other_metrics(metrics_path) if metrics_path.exists() else None

                    if stats and metrics:
                        row["status"] = "ok"
                        row.update(stats)
                        row.update(metrics)
                    elif stats and not metrics:
                        row["status"] = "missing_metrics"
                        row.update(stats)
                    elif metrics and not stats:
                        row["status"] = "missing_stats"
                        row.update(metrics)
                    else:
                        row["status"] = "missing_both"

                    rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tartanair_sweep_v1.parquet"))
    args = parser.parse_args()

    df = collect_all(args.data_root)
    df.to_parquet(args.output, index=False)

    # Summary
    print(f"Total rows: {len(df)}")
    print(f"Status counts:\n{df['status'].value_counts().to_string()}")
    print(f"Envs: {df['env'].nunique()}, Presets: {df['preset'].nunique()}, "
          f"Sequences: {df['sequence'].nunique()}")