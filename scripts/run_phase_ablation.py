from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from pose_recognization.phase_common import (
        PHASES,
        build_person_uid,
        load_recommended_features,
        make_group_splits,
        make_kfold_splits,
    )
except ModuleNotFoundError:
    from phase_common import (
        PHASES,
        build_person_uid,
        load_recommended_features,
        make_group_splits,
        make_kfold_splits,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase-aware ablation studies.")
    parser.add_argument("--features-csv", default="log/phase_features.csv")
    parser.add_argument("--feature-set-json", default="log/feature_diag_person_uid/recommended_feature_set.json")
    parser.add_argument("--target-col", default="distance_cm")
    parser.add_argument("--group-col", default="person_uid")
    parser.add_argument("--min-phase-conf", type=float, default=0.25)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--out-dir", default="log/phase_ablation")
    return parser.parse_args()


def build_non_phase_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    cols: List[str] = []
    metrics = [
        "arm_raise",
        "leg_forward",
        "knee_angle",
        "hip_angle",
        "trunk_lean_deg",
        "kps_conf_mean",
        "kps_valid_ratio",
    ]
    for metric in metrics:
        ph_cols = [f"{ph}_{metric}" for ph in PHASES if f"{ph}_{metric}" in out.columns]
        if not ph_cols:
            continue
        vals = out[ph_cols].apply(pd.to_numeric, errors="coerce")
        for stat_name, series in (
            ("mean", vals.mean(axis=1)),
            ("std", vals.std(axis=1, ddof=0)),
            ("min", vals.min(axis=1)),
            ("max", vals.max(axis=1)),
        ):
            col = f"agg_{metric}_{stat_name}"
            out[col] = series
            cols.append(col)

    for c in [
        "height_cm",
        "weight_kg",
        "prep_frames",
        "takeoff_to_peak_frames",
        "peak_to_landing_frames",
        "flight_frames",
    ]:
        if c in out.columns:
            cols.append(c)
    return out, cols


def eval_config(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    groups: np.ndarray,
    cv_splits: int,
    seed: int,
    rf_n_estimators: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if len(features) == 0:
        return rows
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(y)
    for c in features:
        mask &= np.isfinite(X_df[c].to_numpy(dtype=float))
    mask &= pd.Series(groups).notna().to_numpy()

    if int(np.sum(mask)) < 20:
        return rows

    X = X_df.loc[mask, features].to_numpy(dtype=float)
    y = y[mask]
    g = groups[mask]
    n_groups = len(set(g.tolist()))
    if n_groups < 2:
        return rows
    gk_splits = make_group_splits(X, y, g, cv_splits)
    kf_splits = make_kfold_splits(X, y, cv_splits, seed)

    models = [
        ("Ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        (
            "RandomForest",
            lambda: RandomForestRegressor(n_estimators=rf_n_estimators, random_state=seed, n_jobs=-1),
        ),
    ]
    for model_name, model_factory in models:
        for cv_name, splits in [("GroupKFold", gk_splits), ("KFold", kf_splits)]:
            maes, rmses, r2s, biases = [], [], [], []
            for tr, va in splits:
                model = model_factory()
                model.fit(X[tr], y[tr])
                pred = model.predict(X[va])
                maes.append(float(mean_absolute_error(y[va], pred)))
                rmses.append(float(np.sqrt(mean_squared_error(y[va], pred))))
                r2s.append(float(r2_score(y[va], pred)) if len(va) > 1 else float("nan"))
                biases.append(float(np.mean(pred - y[va])))
            rows.append(
                {
                    "model": model_name,
                    "cv": cv_name,
                    "n_features": int(len(features)),
                    "usable_rows": int(len(y)),
                    "group_count": int(n_groups),
                    "mae_mean": float(np.mean(maes)),
                    "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
                    "r2_mean": float(np.mean(r2s)),
                    "r2_std": float(np.std(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
                    "bias_mean": float(np.mean(biases)),
                    "bias_std": float(np.std(biases, ddof=1)) if len(biases) > 1 else 0.0,
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
    groups_all = df[args.group_col].astype(str).to_numpy()

    rec_features = load_recommended_features(df, args.feature_set_json, fallback_k=15)
    if not rec_features:
        raise SystemExit("No recommended features for ablation.")

    df_np, non_phase_full = build_non_phase_features(df)

    has_idx = df["has_all_indices"].astype(bool) if "has_all_indices" in df.columns else pd.Series([True] * len(df))
    has_kps = df["has_all_kps_len25"].astype(bool) if "has_all_kps_len25" in df.columns else pd.Series([True] * len(df))
    min_conf = pd.to_numeric(df["min_phase_conf"], errors="coerce") if "min_phase_conf" in df.columns else pd.Series([1.0] * len(df))
    qc_mask = has_idx & has_kps & np.isfinite(min_conf) & (min_conf >= float(args.min_phase_conf))
    no_qc_mask = pd.Series([True] * len(df))

    upper_features = [f for f in rec_features if "arm_raise" in f]
    lower_features = [f for f in rec_features if any(k in f for k in ("leg_forward", "knee_angle", "hip_angle"))]
    trunk_features = [f for f in rec_features if "trunk_lean" in f]
    timing_features = [f for f in ["prep_frames", "takeoff_to_peak_frames", "peak_to_landing_frames", "flight_frames"] if f in rec_features]

    configs = [
        ("phase_aligned_qc_full", df.loc[qc_mask].copy(), rec_features),
        ("phase_aligned_no_qc_full", df.loc[no_qc_mask].copy(), rec_features),
        ("non_phase_qc_full", df_np.loc[qc_mask].copy(), non_phase_full),
        ("non_phase_no_qc_full", df_np.loc[no_qc_mask].copy(), non_phase_full),
        ("phase_aligned_qc_upper", df.loc[qc_mask].copy(), upper_features),
        ("phase_aligned_qc_lower", df.loc[qc_mask].copy(), lower_features),
        ("phase_aligned_qc_trunk", df.loc[qc_mask].copy(), trunk_features),
        ("phase_aligned_qc_timing", df.loc[qc_mask].copy(), timing_features),
    ]

    rows: List[Dict[str, object]] = []
    for cfg_name, cfg_df, cfg_features in configs:
        cfg_groups = cfg_df[args.group_col].astype(str).to_numpy()
        metrics_rows = eval_config(
            df=cfg_df,
            features=[c for c in cfg_features if c in cfg_df.columns],
            target_col=args.target_col,
            groups=cfg_groups,
            cv_splits=args.cv_splits,
            seed=args.seed,
            rf_n_estimators=args.rf_n_estimators,
        )
        for row in metrics_rows:
            row["config"] = cfg_name
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    if len(summary_df) == 0:
        raise SystemExit("No ablation results generated. Check data size and feature availability.")
    summary_df = summary_df.sort_values(["cv", "mae_mean", "config", "model"]).reset_index(drop=True)

    out_summary = out_dir / "phase_ablation_summary.csv"
    summary_df.to_csv(out_summary, index=False, encoding="utf-8")

    # Key contrasts aligned with paper claims.
    def pick_metric(config: str, model: str, cv: str) -> Dict[str, float]:
        sub = summary_df[
            (summary_df["config"] == config) & (summary_df["model"] == model) & (summary_df["cv"] == cv)
        ]
        if len(sub) == 0:
            return {"mae_mean": float("nan"), "r2_mean": float("nan")}
        r = sub.iloc[0]
        return {"mae_mean": float(r["mae_mean"]), "r2_mean": float(r["r2_mean"])}

    phase_qc = pick_metric("phase_aligned_qc_full", "RandomForest", "GroupKFold")
    nonphase_qc = pick_metric("non_phase_qc_full", "RandomForest", "GroupKFold")
    phase_noqc = pick_metric("phase_aligned_no_qc_full", "RandomForest", "GroupKFold")
    key_payload = {
        "phase_vs_nonphase_qc": {
            "delta_mae_nonphase_minus_phase": float(nonphase_qc["mae_mean"] - phase_qc["mae_mean"]),
            "delta_r2_phase_minus_nonphase": float(phase_qc["r2_mean"] - nonphase_qc["r2_mean"]),
        },
        "qc_gate_effect_phase_aligned": {
            "delta_mae_noqc_minus_qc": float(phase_noqc["mae_mean"] - phase_qc["mae_mean"]),
            "delta_r2_qc_minus_noqc": float(phase_qc["r2_mean"] - phase_noqc["r2_mean"]),
        },
    }
    out_key = out_dir / "phase_ablation_key_contrasts.json"
    out_key.write_text(json.dumps(key_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {out_summary}")
    print(f"saved: {out_key}")
    print("\n[Ablation summary top rows]")
    print(summary_df.head(16).to_string(index=False))


if __name__ == "__main__":
    main()
