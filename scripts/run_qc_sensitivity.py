from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Evaluate QC threshold sensitivity.")
    parser.add_argument("--features-csv", default="log/phase_features.csv")
    parser.add_argument("--feature-set-json", default="log/feature_diag_person_uid/recommended_feature_set.json")
    parser.add_argument("--target-col", default="distance_cm")
    parser.add_argument("--group-col", default="person_uid")
    parser.add_argument("--thresholds", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--out-dir", default="log/qc_sensitivity")
    return parser.parse_args()


def parse_thresholds(text: str) -> List[float]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return sorted(set(vals))


def eval_group_cv_rf(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    groups: np.ndarray,
    cv_splits: int,
    seed: int,
    n_estimators: int,
) -> Dict[str, float]:
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    mask = np.isfinite(y)
    for c in feature_cols:
        mask &= np.isfinite(X_df[c].to_numpy(dtype=float))
    mask &= pd.Series(groups).notna().to_numpy()
    if int(np.sum(mask)) < 20:
        return {
            "usable_rows": int(np.sum(mask)),
            "group_count": 0,
            "mae_mean": float("nan"),
            "rmse_mean": float("nan"),
            "r2_mean": float("nan"),
            "bias_mean": float("nan"),
        }

    X = X_df.loc[mask, feature_cols].to_numpy(dtype=float)
    y = y[mask]
    g = groups[mask]
    n_groups = len(set(g.tolist()))
    if n_groups < 2:
        return {
            "usable_rows": int(len(y)),
            "group_count": int(n_groups),
            "mae_mean": float("nan"),
            "rmse_mean": float("nan"),
            "r2_mean": float("nan"),
            "bias_mean": float("nan"),
        }
    splits = make_group_splits(X, y, g, cv_splits)

    maes, rmses, r2s, biases = [], [], [], []
    for tr, va in splits:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        maes.append(float(mean_absolute_error(y[va], pred)))
        rmses.append(float(np.sqrt(mean_squared_error(y[va], pred))))
        r2s.append(float(r2_score(y[va], pred)) if len(va) > 1 else float("nan"))
        biases.append(float(np.mean(pred - y[va])))

    return {
        "usable_rows": int(len(y)),
        "group_count": int(n_groups),
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

    feature_cols = load_recommended_features(df, args.feature_set_json, fallback_k=15)
    if not feature_cols:
        raise SystemExit("No recommended features for QC sensitivity.")

    thresholds = parse_thresholds(args.thresholds)
    groups_all = df[args.group_col].astype(str).to_numpy()
    n_total = len(df)

    base_gender_m_ratio = float((df["gender"].astype(str) == "M").mean()) if "gender" in df.columns else float("nan")
    base_height_mean = float(pd.to_numeric(df["height_cm"], errors="coerce").mean()) if "height_cm" in df.columns else float("nan")
    base_weight_mean = float(pd.to_numeric(df["weight_kg"], errors="coerce").mean()) if "weight_kg" in df.columns else float("nan")
    base_dist_mean = float(pd.to_numeric(df[args.target_col], errors="coerce").mean())

    has_idx = df["has_all_indices"].astype(bool) if "has_all_indices" in df.columns else pd.Series([True] * len(df))
    has_kps = df["has_all_kps_len25"].astype(bool) if "has_all_kps_len25" in df.columns else pd.Series([True] * len(df))
    min_conf = pd.to_numeric(df["min_phase_conf"], errors="coerce") if "min_phase_conf" in df.columns else pd.Series([1.0] * len(df))

    rows: List[Dict[str, float]] = []

    # no-QC baseline
    no_qc_mask = pd.Series([True] * len(df))
    no_qc_stats = eval_group_cv_rf(
        df=df.loc[no_qc_mask].copy(),
        feature_cols=feature_cols,
        target_col=args.target_col,
        groups=groups_all[no_qc_mask.to_numpy()],
        cv_splits=args.cv_splits,
        seed=args.seed,
        n_estimators=args.rf_n_estimators,
    )
    rows.append(
        {
            "condition": "no_qc_gate",
            "min_phase_conf_thr": float("nan"),
            "n_kept": int(np.sum(no_qc_mask)),
            "keep_ratio": float(np.sum(no_qc_mask) / max(n_total, 1)),
            "low_conf_rate": float("nan"),
            "missing_indices_rate": float(np.mean(~has_idx)),
            "invalid_kps_rate": float(np.mean(~has_kps)),
            "gender_m_ratio": base_gender_m_ratio,
            "delta_gender_m_ratio": 0.0,
            "delta_height_mean": 0.0,
            "delta_weight_mean": 0.0,
            "delta_distance_mean": 0.0,
            **no_qc_stats,
        }
    )

    for thr in thresholds:
        gate_mask = has_idx & has_kps & np.isfinite(min_conf) & (min_conf >= float(thr))
        sub = df.loc[gate_mask].copy()
        n_kept = int(len(sub))
        keep_ratio = float(n_kept / max(n_total, 1))
        if n_kept == 0:
            rows.append(
                {
                    "condition": "qc_gate",
                    "min_phase_conf_thr": float(thr),
                    "n_kept": 0,
                    "keep_ratio": 0.0,
                    "low_conf_rate": float(np.mean(np.isfinite(min_conf) & (min_conf < thr))),
                    "missing_indices_rate": float(np.mean(~has_idx)),
                    "invalid_kps_rate": float(np.mean(~has_kps)),
                    "gender_m_ratio": float("nan"),
                    "delta_gender_m_ratio": float("nan"),
                    "delta_height_mean": float("nan"),
                    "delta_weight_mean": float("nan"),
                    "delta_distance_mean": float("nan"),
                    "usable_rows": 0,
                    "group_count": 0,
                    "mae_mean": float("nan"),
                    "rmse_mean": float("nan"),
                    "r2_mean": float("nan"),
                    "bias_mean": float("nan"),
                }
            )
            continue

        gender_m = float((sub["gender"].astype(str) == "M").mean()) if "gender" in sub.columns else float("nan")
        h_mean = float(pd.to_numeric(sub["height_cm"], errors="coerce").mean()) if "height_cm" in sub.columns else float("nan")
        w_mean = float(pd.to_numeric(sub["weight_kg"], errors="coerce").mean()) if "weight_kg" in sub.columns else float("nan")
        d_mean = float(pd.to_numeric(sub[args.target_col], errors="coerce").mean())
        cv_stats = eval_group_cv_rf(
            df=sub,
            feature_cols=feature_cols,
            target_col=args.target_col,
            groups=groups_all[gate_mask.to_numpy()],
            cv_splits=args.cv_splits,
            seed=args.seed,
            n_estimators=args.rf_n_estimators,
        )
        rows.append(
            {
                "condition": "qc_gate",
                "min_phase_conf_thr": float(thr),
                "n_kept": n_kept,
                "keep_ratio": keep_ratio,
                "low_conf_rate": float(np.mean(np.isfinite(min_conf) & (min_conf < thr))),
                "missing_indices_rate": float(np.mean(~has_idx)),
                "invalid_kps_rate": float(np.mean(~has_kps)),
                "gender_m_ratio": gender_m,
                "delta_gender_m_ratio": gender_m - base_gender_m_ratio,
                "delta_height_mean": h_mean - base_height_mean,
                "delta_weight_mean": w_mean - base_weight_mean,
                "delta_distance_mean": d_mean - base_dist_mean,
                **cv_stats,
            }
        )

    out_df = pd.DataFrame(rows)
    summary_path = out_dir / "qc_threshold_sensitivity.csv"
    summary_path.write_text(out_df.to_csv(index=False), encoding="utf-8")

    # Recommend threshold by lowest MAE under keep_ratio >= 0.6
    qc_rows = out_df[(out_df["condition"] == "qc_gate") & (out_df["keep_ratio"] >= 0.60)].copy()
    if len(qc_rows) > 0 and qc_rows["mae_mean"].notna().any():
        best = qc_rows.sort_values("mae_mean", ascending=True).iloc[0]
        rec = {
            "recommended_min_phase_conf": float(best["min_phase_conf_thr"]),
            "mae_mean": float(best["mae_mean"]),
            "r2_mean": float(best["r2_mean"]),
            "keep_ratio": float(best["keep_ratio"]),
            "n_kept": int(best["n_kept"]),
        }
    else:
        rec = {}

    global_stats = {
        "n_total": int(n_total),
        "missing_indices_rate": float(np.mean(~has_idx)),
        "invalid_kps_rate": float(np.mean(~has_kps)),
        "base_gender_m_ratio": base_gender_m_ratio,
        "base_height_mean": base_height_mean,
        "base_weight_mean": base_weight_mean,
        "base_distance_mean": base_dist_mean,
        "recommended_threshold": rec,
    }
    stats_path = out_dir / "qc_failure_and_bias_stats.json"
    stats_path.write_text(json.dumps(global_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {summary_path}")
    print(f"saved: {stats_path}")
    print("\n[QC threshold summary top rows]")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
