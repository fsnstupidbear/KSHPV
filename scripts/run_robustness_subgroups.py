from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    parser = argparse.ArgumentParser(description="Run subgroup robustness analysis on OOF predictions.")
    parser.add_argument("--features-csv", default="log/phase_features_clean_filtered_v2.csv")
    parser.add_argument("--feature-set-json", default="log/feature_diag_person_uid/recommended_feature_set.json")
    parser.add_argument("--target-col", default="distance_cm")
    parser.add_argument("--group-col", default="person_uid")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-iter", type=int, default=800)
    parser.add_argument("--out-dir", default="log/robustness_subgroups")
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    sims = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, len(vals), size=len(vals))
        sims[i] = float(np.mean(vals[idx]))
    return float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))


def run_oof_rf(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_splits: int,
    seed: int,
    n_estimators: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    splits = make_group_splits(X, y, groups, cv_splits)
    oof = np.full(len(y), np.nan, dtype=float)
    rows: List[Dict[str, float]] = []
    for fold_id, (tr, va) in enumerate(splits, start=1):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        oof[va] = pred
        rows.append(
            {
                "fold": float(fold_id),
                "n_val": float(len(va)),
                "mae": float(mean_absolute_error(y[va], pred)),
                "rmse": float(np.sqrt(mean_squared_error(y[va], pred))),
                "r2": float(r2_score(y[va], pred)) if len(va) > 1 else float("nan"),
                "bias": float(np.mean(pred - y[va])),
            }
        )
    return oof, pd.DataFrame(rows)


def add_subgroup_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    if {"height_cm", "weight_kg"}.issubset(out.columns):
        h_m = pd.to_numeric(out["height_cm"], errors="coerce") / 100.0
        w = pd.to_numeric(out["weight_kg"], errors="coerce")
        bmi = w / (h_m * h_m)
        out["bmi"] = bmi
        out["bmi_group"] = pd.cut(
            bmi,
            bins=[-np.inf, 18.5, 24.0, np.inf],
            labels=["under", "normal", "over"],
        ).astype(str)
    else:
        out["bmi"] = np.nan
        out["bmi_group"] = "unknown"

    y = pd.to_numeric(out[target_col], errors="coerce")
    q1, q2 = y.quantile([1.0 / 3.0, 2.0 / 3.0])
    out["distance_level"] = pd.cut(
        y,
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "mid", "high"],
    ).astype(str)
    return out


def subgroup_metric_rows(
    df: pd.DataFrame,
    subgroup_col: str,
    bootstrap_iter: int,
    seed: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    levels = sorted(df[subgroup_col].dropna().astype(str).unique().tolist())
    for i, level in enumerate(levels):
        sub = df[df[subgroup_col].astype(str) == level].copy()
        if len(sub) < 5:
            continue
        abs_err = np.abs(sub["error"].to_numpy(dtype=float))
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(np.square(sub["error"].to_numpy(dtype=float)))))
        bias = float(np.mean(sub["error"].to_numpy(dtype=float)))
        r2 = float(r2_score(sub["y_true"], sub["y_pred"])) if len(sub) > 1 else float("nan")
        ci_low, ci_high = bootstrap_ci(abs_err, bootstrap_iter, seed + i)
        rows.append(
            {
                "subgroup_col": subgroup_col,
                "subgroup_level": level,
                "n": int(len(sub)),
                "mae": mae,
                "mae_ci_low": ci_low,
                "mae_ci_high": ci_high,
                "rmse": rmse,
                "bias": bias,
                "r2": r2,
            }
        )
    return rows


def subgroup_test_rows(df: pd.DataFrame, subgroup_col: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    vals = []
    labels = []
    for level in sorted(df[subgroup_col].dropna().astype(str).unique().tolist()):
        arr = np.abs(df.loc[df[subgroup_col].astype(str) == level, "error"].to_numpy(dtype=float))
        arr = arr[np.isfinite(arr)]
        if len(arr) >= 5:
            vals.append(arr)
            labels.append(level)
    if len(vals) < 2:
        return rows

    if len(vals) == 2:
        u = stats.mannwhitneyu(vals[0], vals[1], alternative="two-sided")
        rows.append(
            {
                "subgroup_col": subgroup_col,
                "test": "mannwhitneyu",
                "groups": f"{labels[0]} vs {labels[1]}",
                "statistic": float(u.statistic),
                "p_value": float(u.pvalue),
            }
        )
    else:
        kw = stats.kruskal(*vals)
        rows.append(
            {
                "subgroup_col": subgroup_col,
                "test": "kruskal",
                "groups": ",".join(labels),
                "statistic": float(kw.statistic),
                "p_value": float(kw.pvalue),
            }
        )
    return rows


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
        raise SystemExit("No recommended features for subgroup robustness.")

    y = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(dtype=float)
    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    groups = df[args.group_col].astype(str).to_numpy()
    mask = np.isfinite(y)
    for c in features:
        mask &= np.isfinite(X_df[c].to_numpy(dtype=float))
    mask &= pd.Series(groups).notna().to_numpy()

    X = X_df.loc[mask, features].to_numpy(dtype=float)
    y = y[mask]
    g = groups[mask]
    meta = df.loc[mask].copy().reset_index(drop=True)

    oof_pred, fold_df = run_oof_rf(
        X=X,
        y=y,
        groups=g,
        cv_splits=args.cv_splits,
        seed=args.seed,
        n_estimators=args.rf_n_estimators,
    )
    oof_df = meta.copy()
    oof_df["y_true"] = y
    oof_df["y_pred"] = oof_pred
    oof_df["error"] = oof_df["y_pred"] - oof_df["y_true"]
    oof_df = add_subgroup_columns(oof_df, args.target_col)

    metric_rows: List[Dict[str, float]] = []
    test_rows: List[Dict[str, float]] = []
    for col in ["gender", "bmi_group", "distance_level"]:
        if col in oof_df.columns:
            metric_rows.extend(
                subgroup_metric_rows(oof_df, subgroup_col=col, bootstrap_iter=args.bootstrap_iter, seed=args.seed)
            )
            test_rows.extend(subgroup_test_rows(oof_df, subgroup_col=col))

    metric_df = pd.DataFrame(metric_rows)
    test_df = pd.DataFrame(test_rows)

    overall = {
        "n": int(len(oof_df)),
        "mae": float(np.mean(np.abs(oof_df["error"]))),
        "rmse": float(np.sqrt(np.mean(np.square(oof_df["error"])))),
        "bias": float(np.mean(oof_df["error"])),
        "r2": float(r2_score(oof_df["y_true"], oof_df["y_pred"])) if len(oof_df) > 1 else float("nan"),
    }
    summary_payload = {
        "overall_oof": overall,
        "fold_mae_mean": float(fold_df["mae"].mean()),
        "fold_r2_mean": float(fold_df["r2"].mean()),
        "n_subgroup_rows": int(len(metric_df)),
        "n_tests": int(len(test_df)),
    }

    oof_path = out_dir / "oof_predictions_with_subgroups.csv"
    fold_path = out_dir / "cv_folds_main_rf.csv"
    metric_path = out_dir / "subgroup_metrics.csv"
    test_path = out_dir / "subgroup_error_tests.csv"
    summary_path = out_dir / "subgroup_robustness_summary.json"

    oof_df.to_csv(oof_path, index=False, encoding="utf-8")
    fold_df.to_csv(fold_path, index=False, encoding="utf-8")
    metric_df.to_csv(metric_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {oof_path}")
    print(f"saved: {fold_path}")
    print(f"saved: {metric_path}")
    print(f"saved: {test_path}")
    print(f"saved: {summary_path}")
    print("\n[Overall OOF]")
    print(summary_payload["overall_oof"])


if __name__ == "__main__":
    main()
