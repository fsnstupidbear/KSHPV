from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from pose_recognization.phase_common import (
        build_person_uid,
        load_recommended_features,
        make_group_splits,
    )
except ModuleNotFoundError:
    from phase_common import (
        build_person_uid,
        load_recommended_features,
        make_group_splits,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Noise and missingness sensitivity on phase-aware model.")
    parser.add_argument("--features-csv", default="log/phase_features_clean_filtered_v2.csv")
    parser.add_argument("--feature-set-json", default="log/feature_diag_person_uid/recommended_feature_set.json")
    parser.add_argument("--target-col", default="distance_cm")
    parser.add_argument("--group-col", default="person_uid")
    parser.add_argument("--noise-levels", default="0,0.02,0.05,0.10")
    parser.add_argument("--missing-rates", default="0,0.05,0.10,0.20")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--out-dir", default="log/noise_missing_sensitivity")
    return parser.parse_args()


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return sorted(set(out))


def run_eval(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_splits: int,
    seed: int,
    n_estimators: int,
) -> Dict[str, float]:
    splits = make_group_splits(X, y, groups, cv_splits)
    maes, rmses, r2s, biases = [], [], [], []
    for tr, va in splits:
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        maes.append(float(mean_absolute_error(y[va], pred)))
        rmses.append(float(np.sqrt(mean_squared_error(y[va], pred))))
        r2s.append(float(r2_score(y[va], pred)) if len(va) > 1 else float("nan"))
        biases.append(float(np.mean(pred - y[va])))
    return {
        "mae_mean": float(np.mean(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "r2_mean": float(np.mean(r2s)),
        "bias_mean": float(np.mean(biases)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    if args.group_col == "person_uid" and "person_uid" not in df.columns:
        df["person_uid"] = build_person_uid(df)
    if args.group_col not in df.columns:
        raise SystemExit(f"group_col not found: {args.group_col}")

    features = load_recommended_features(df, args.feature_set_json, fallback_k=15)
    if not features:
        raise SystemExit("No recommended features for sensitivity test.")

    y = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(dtype=float)
    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    groups = df[args.group_col].astype(str).to_numpy()
    mask = np.isfinite(y)
    for c in features:
        mask &= np.isfinite(X_df[c].to_numpy(dtype=float))
    mask &= pd.Series(groups).notna().to_numpy()
    if int(np.sum(mask)) < 20:
        raise SystemExit("Too few valid rows for sensitivity analysis.")

    X = X_df.loc[mask, features].to_numpy(dtype=float)
    y = y[mask]
    g = groups[mask]
    col_std = np.std(X, axis=0, ddof=0)
    col_std[col_std < 1e-12] = 1.0

    noise_levels = parse_float_list(args.noise_levels)
    missing_rates = parse_float_list(args.missing_rates)
    rows: List[Dict[str, float]] = []

    for noise_level in noise_levels:
        for missing_rate in missing_rates:
            for rep in range(args.repeats):
                rng = np.random.RandomState(args.seed + rep * 1000 + int(noise_level * 10000) + int(missing_rate * 1000))
                Xp = X.copy()
                if noise_level > 0:
                    noise = rng.normal(loc=0.0, scale=noise_level, size=Xp.shape) * col_std.reshape(1, -1)
                    Xp = Xp + noise
                if missing_rate > 0:
                    miss_mask = rng.rand(*Xp.shape) < missing_rate
                    Xp[miss_mask] = np.nan

                stats_row = run_eval(
                    X=Xp,
                    y=y,
                    groups=g,
                    cv_splits=args.cv_splits,
                    seed=args.seed,
                    n_estimators=args.rf_n_estimators,
                )
                rows.append(
                    {
                        "noise_level": float(noise_level),
                        "missing_rate": float(missing_rate),
                        "repeat": int(rep),
                        "usable_rows": int(len(y)),
                        "group_count": int(len(set(g.tolist()))),
                        **stats_row,
                    }
                )

    raw_df = pd.DataFrame(rows)
    summary_df = (
        raw_df.groupby(["noise_level", "missing_rate"], as_index=False)
        .agg(
            runs=("repeat", "count"),
            mae_mean=("mae_mean", "mean"),
            mae_std=("mae_mean", "std"),
            rmse_mean=("rmse_mean", "mean"),
            r2_mean=("r2_mean", "mean"),
            r2_std=("r2_mean", "std"),
            bias_mean=("bias_mean", "mean"),
        )
        .sort_values(["noise_level", "missing_rate"])
        .reset_index(drop=True)
    )

    baseline = summary_df[(summary_df["noise_level"] == 0.0) & (summary_df["missing_rate"] == 0.0)]
    if len(baseline) > 0:
        b_mae = float(baseline.iloc[0]["mae_mean"])
        b_r2 = float(baseline.iloc[0]["r2_mean"])
        summary_df["delta_mae_vs_baseline"] = summary_df["mae_mean"] - b_mae
        summary_df["delta_r2_vs_baseline"] = summary_df["r2_mean"] - b_r2
    else:
        summary_df["delta_mae_vs_baseline"] = np.nan
        summary_df["delta_r2_vs_baseline"] = np.nan

    raw_path = out_dir / "noise_missing_raw_runs.csv"
    sum_path = out_dir / "noise_missing_summary.csv"
    json_path = out_dir / "noise_missing_summary.json"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8")
    summary_df.to_csv(sum_path, index=False, encoding="utf-8")

    payload = {
        "features_csv": args.features_csv,
        "feature_set_json": args.feature_set_json,
        "noise_levels": noise_levels,
        "missing_rates": missing_rates,
        "repeats": args.repeats,
        "best_condition_by_mae": summary_df.sort_values("mae_mean", ascending=True).head(1).to_dict(orient="records"),
        "worst_condition_by_mae": summary_df.sort_values("mae_mean", ascending=False).head(1).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {raw_path}")
    print(f"saved: {sum_path}")
    print(f"saved: {json_path}")
    print("\n[Noise/missing summary top rows]")
    print(summary_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
