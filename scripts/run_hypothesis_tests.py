from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from pose_recognization.phase_common import PHASES, build_person_uid, load_recommended_features, make_group_splits
except ModuleNotFoundError:
    from phase_common import PHASES, build_person_uid, load_recommended_features, make_group_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hypothesis-driven tests for phase-aware jump analysis.")
    parser.add_argument("--features-csv", default="log/phase_features_clean_filtered_v2.csv")
    parser.add_argument("--feature-set-json", default="log/feature_diag_person_uid/recommended_feature_set.json")
    parser.add_argument("--target-col", default="distance_cm")
    parser.add_argument("--group-col", default="person_uid")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--h1-splits", type=int, default=30)
    parser.add_argument("--h1-test-size", type=float, default=0.20)
    parser.add_argument("--h2-splits", type=int, default=30)
    parser.add_argument("--h2-test-size", type=float, default=0.20)
    parser.add_argument("--bootstrap-iter", type=int, default=800)
    parser.add_argument("--perm-iter", type=int, default=2000)
    parser.add_argument("--out-dir", default="log/hypothesis_tests")
    return parser.parse_args()


def build_h1_feature_sets(df: pd.DataFrame, rec_features: Sequence[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    H1 compares:
    - phase-aligned kinematic features (selected features with phase/time tags)
    - coarse non-phase kinematic baselines (global phase-mean per metric family)
    This avoids inflating non-phase baseline dimensionality with mean/std/min/max blocks.
    """
    out = df.copy()

    # Keep kinematic families only, excluding anthropometrics/timing/QC meta columns.
    kin_tokens = ("arm_raise", "leg_forward", "knee_angle", "hip_angle", "trunk_lean_deg")
    phase_kin_features = [f for f in rec_features if any(tok in f for tok in kin_tokens)]

    metric_families = ["arm_raise", "leg_forward", "knee_angle", "hip_angle", "trunk_lean_deg"]
    non_phase_coarse: List[str] = []
    for metric in metric_families:
        cols = [f"{ph}_{metric}" for ph in PHASES if f"{ph}_{metric}" in out.columns]
        if not cols:
            continue
        col = f"agg_{metric}_mean"
        out[col] = out[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        non_phase_coarse.append(col)

    return out, phase_kin_features, non_phase_coarse


def make_group_shuffle_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    test_size: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_groups = len(set(groups.tolist()))
    if n_groups < 3:
        # Fallback to leave enough train/val groups.
        return make_group_splits(X, y, groups, min(2, n_groups))
    gss = GroupShuffleSplit(
        n_splits=max(3, int(n_splits)),
        test_size=float(test_size),
        random_state=seed,
    )
    return list(gss.split(X, y, groups=groups))


def ensure_group(df: pd.DataFrame, group_col: str) -> np.ndarray:
    key = str(group_col).strip().lower()
    if key == "person_uid":
        if "person_uid" in df.columns:
            return df["person_uid"].astype(str).to_numpy()
        return build_person_uid(df).astype(str).to_numpy()
    if group_col not in df.columns:
        raise ValueError(f"group_col not found: {group_col}")
    return df[group_col].astype(str).to_numpy()


def pick_coordination_features(df: pd.DataFrame, rec_features: Sequence[str]) -> List[str]:
    """
    Predefined coordination block for H2.
    Priority is biomechanical coordination signals; we allow features not in rec_features
    so H2 tests a standalone scientific block rather than a feature-selection artifact.
    """
    priority = [
        "arm_raise_takeoff_to_peak",
        "knee_ext_takeoff_to_peak",
        "takeoff_mid_arm_raise",
        "peak_arm_raise",
        "takeoff_mid_leg_forward",
        "takeoff_to_peak_frames",
    ]
    out = [c for c in priority if c in df.columns]
    # Fallback: use any recommended coordination-like signals.
    if len(out) < 2:
        tokens = ("arm_raise", "knee_ext", "coord", "takeoff_to_peak")
        out = [c for c in rec_features if c in df.columns and any(t in c for t in tokens)]
    return list(dict.fromkeys(out))


def bootstrap_ci_mean(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    means = np.zeros(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, len(vals), size=len(vals))
        means[i] = float(np.mean(vals[idx]))
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def paired_sign_permutation_p(
    diffs: np.ndarray, n_iter: int, seed: int, alternative: str = "greater"
) -> float:
    rng = np.random.RandomState(seed)
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    obs = float(np.mean(d))
    if len(d) == 1:
        if alternative == "greater":
            return 1.0 if obs <= 0 else 0.5
        if alternative == "less":
            return 1.0 if obs >= 0 else 0.5
        return 1.0

    sims = np.zeros(n_iter, dtype=float)
    signs = np.array([-1.0, 1.0], dtype=float)
    for i in range(n_iter):
        s = rng.choice(signs, size=len(d), replace=True)
        sims[i] = float(np.mean(d * s))

    if alternative == "greater":
        p = (np.sum(sims >= obs) + 1.0) / (n_iter + 1.0)
    elif alternative == "less":
        p = (np.sum(sims <= obs) + 1.0) / (n_iter + 1.0)
    else:
        p = (np.sum(np.abs(sims) >= abs(obs)) + 1.0) / (n_iter + 1.0)
    return float(p)


def holm_adjust_pvalues(pvals: Sequence[float]) -> List[float]:
    vals = np.asarray(pvals, dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    valid_idx = np.where(np.isfinite(vals))[0]
    if len(valid_idx) == 0:
        return out.tolist()
    order = valid_idx[np.argsort(vals[valid_idx])]
    adj_tmp = np.zeros(len(order), dtype=float)
    for i, idx in enumerate(order):
        adj_tmp[i] = min((len(order) - i) * float(vals[idx]), 1.0)
    # Holm monotonicity
    for i in range(1, len(adj_tmp)):
        adj_tmp[i] = max(adj_tmp[i], adj_tmp[i - 1])
    for i, idx in enumerate(order):
        out[idx] = adj_tmp[i]
    return out.tolist()


def run_fold_eval(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    model_factory,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rows: List[Dict[str, float]] = []
    oof = np.full(len(y), np.nan, dtype=float)
    for fold_id, (tr, va) in enumerate(splits, start=1):
        model = model_factory()
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
    return pd.DataFrame(rows), oof


def residual_partial_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    conf_cols: Sequence[str],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    cols = [x_col, y_col] + list(conf_cols)
    work = df[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    if len(work) < 20:
        return {
            "feature": x_col,
            "n": int(len(work)),
            "rho": float("nan"),
            "p_two_sided": float("nan"),
            "rho_ci_low": float("nan"),
            "rho_ci_high": float("nan"),
        }

    Xc = work[list(conf_cols)].to_numpy(dtype=float)
    xv = work[x_col].to_numpy(dtype=float)
    yv = work[y_col].to_numpy(dtype=float)
    reg_x = LinearRegression().fit(Xc, xv)
    reg_y = LinearRegression().fit(Xc, yv)
    rx = xv - reg_x.predict(Xc)
    ry = yv - reg_y.predict(Xc)
    rho, p = stats.spearmanr(rx, ry)
    rho = float(rho) if np.isfinite(rho) else float("nan")
    p = float(p) if np.isfinite(p) else float("nan")

    rng = np.random.RandomState(seed)
    boot = np.zeros(n_boot, dtype=float)
    idx_all = np.arange(len(rx))
    for i in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        brho, _ = stats.spearmanr(rx[idx], ry[idx])
        boot[i] = float(brho) if np.isfinite(brho) else 0.0
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    return {
        "feature": x_col,
        "n": int(len(work)),
        "rho": rho,
        "p_two_sided": p,
        "rho_ci_low": lo,
        "rho_ci_high": hi,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    rec_features = load_recommended_features(df, args.feature_set_json, fallback_k=15)
    if not rec_features:
        raise SystemExit("No recommended features found.")

    groups_all = ensure_group(df, args.group_col)
    df_h1, h1_phase_cols, h1_non_phase_cols = build_h1_feature_sets(df, rec_features)
    if len(h1_phase_cols) < 3 or len(h1_non_phase_cols) < 3:
        raise SystemExit(
            f"H1 feature setup invalid: phase={len(h1_phase_cols)} non_phase={len(h1_non_phase_cols)}"
        )

    # H1: phase-aware kinematic representation > coarse non-phase kinematic baseline
    mask_h1 = np.isfinite(pd.to_numeric(df_h1[args.target_col], errors="coerce").to_numpy(dtype=float))
    for c in h1_phase_cols + h1_non_phase_cols:
        mask_h1 &= np.isfinite(pd.to_numeric(df_h1[c], errors="coerce").to_numpy(dtype=float))
    mask_h1 &= pd.Series(groups_all).notna().to_numpy()

    df1 = df_h1.loc[mask_h1].copy().reset_index(drop=True)
    y1 = pd.to_numeric(df1[args.target_col], errors="coerce").to_numpy(dtype=float)
    g1 = groups_all[mask_h1]
    X_phase = df1[h1_phase_cols].to_numpy(dtype=float)
    X_non_phase = df1[h1_non_phase_cols].to_numpy(dtype=float)
    splits1 = make_group_shuffle_splits(
        X_phase,
        y1,
        g1,
        n_splits=args.h1_splits,
        test_size=args.h1_test_size,
        seed=args.seed,
    )

    rf_factory = lambda: RandomForestRegressor(
        n_estimators=args.rf_n_estimators,
        random_state=args.seed,
        n_jobs=-1,
    )

    phase_rows: List[Dict[str, float]] = []
    non_phase_rows: List[Dict[str, float]] = []
    pred_rows: List[Dict[str, float]] = []
    for split_id, (tr, va) in enumerate(splits1, start=1):
        m_phase = rf_factory()
        m_non = rf_factory()
        m_phase.fit(X_phase[tr], y1[tr])
        m_non.fit(X_non_phase[tr], y1[tr])
        pred_phase = m_phase.predict(X_phase[va])
        pred_non = m_non.predict(X_non_phase[va])

        phase_rows.append(
            {
                "fold": float(split_id),
                "n_val": float(len(va)),
                "mae": float(mean_absolute_error(y1[va], pred_phase)),
                "rmse": float(np.sqrt(mean_squared_error(y1[va], pred_phase))),
                "r2": float(r2_score(y1[va], pred_phase)) if len(va) > 1 else float("nan"),
                "bias": float(np.mean(pred_phase - y1[va])),
                "set": "phase_aligned_kinematic",
            }
        )
        non_phase_rows.append(
            {
                "fold": float(split_id),
                "n_val": float(len(va)),
                "mae": float(mean_absolute_error(y1[va], pred_non)),
                "rmse": float(np.sqrt(mean_squared_error(y1[va], pred_non))),
                "r2": float(r2_score(y1[va], pred_non)) if len(va) > 1 else float("nan"),
                "bias": float(np.mean(pred_non - y1[va])),
                "set": "non_phase_coarse_kinematic",
            }
        )
        for local_i, row_idx in enumerate(va.tolist()):
            pred_rows.append(
                {
                    "fold": int(split_id),
                    "row_index": int(row_idx),
                    "group": str(g1[row_idx]),
                    "y_true": float(y1[row_idx]),
                    "y_pred_phase": float(pred_phase[local_i]),
                    "y_pred_non_phase": float(pred_non[local_i]),
                }
            )

    fold_phase = pd.DataFrame(phase_rows)
    fold_non_phase = pd.DataFrame(non_phase_rows)
    h1_fold = pd.concat([fold_phase, fold_non_phase], ignore_index=True)

    mae_diff = fold_non_phase["mae"].to_numpy(dtype=float) - fold_phase["mae"].to_numpy(dtype=float)
    r2_diff = fold_phase["r2"].to_numpy(dtype=float) - fold_non_phase["r2"].to_numpy(dtype=float)
    mae_t2 = stats.ttest_1samp(mae_diff, popmean=0.0, nan_policy="omit")
    r2_t2 = stats.ttest_1samp(r2_diff, popmean=0.0, nan_policy="omit")
    mae_t_p = (
        float(mae_t2.pvalue / 2.0) if np.isfinite(mae_t2.pvalue) and float(np.mean(mae_diff)) > 0 else
        (float(1.0 - mae_t2.pvalue / 2.0) if np.isfinite(mae_t2.pvalue) else float("nan"))
    )
    r2_t_p = (
        float(r2_t2.pvalue / 2.0) if np.isfinite(r2_t2.pvalue) and float(np.mean(r2_diff)) > 0 else
        (float(1.0 - r2_t2.pvalue / 2.0) if np.isfinite(r2_t2.pvalue) else float("nan"))
    )
    mae_perm_p = paired_sign_permutation_p(mae_diff, args.perm_iter, args.seed, alternative="greater")
    r2_perm_p = paired_sign_permutation_p(r2_diff, args.perm_iter, args.seed + 1, alternative="greater")
    mae_ci = bootstrap_ci_mean(mae_diff, args.bootstrap_iter, args.seed)
    r2_ci = bootstrap_ci_mean(r2_diff, args.bootstrap_iter, args.seed + 1)

    # H2: coordination block adds independent value beyond confounders
    if "gender" in df.columns:
        df["gender_male"] = (df["gender"].astype(str) == "M").astype(float)
        confounders = [c for c in ["height_cm", "weight_kg", "gender_male"] if c in df.columns]
    else:
        confounders = [c for c in ["height_cm", "weight_kg"] if c in df.columns]

    coord_features = pick_coordination_features(df, rec_features)
    if len(coord_features) == 0:
        raise SystemExit("No coordination features found for H2.")
    phase_wo_coord = [f for f in rec_features if f not in coord_features]
    base_cols = list(dict.fromkeys(confounders + phase_wo_coord))
    full_cols = list(dict.fromkeys(base_cols + coord_features))

    mask_h2 = np.isfinite(pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(dtype=float))
    for c in full_cols:
        mask_h2 &= np.isfinite(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
    mask_h2 &= pd.Series(groups_all).notna().to_numpy()
    df2 = df.loc[mask_h2].copy()
    y2 = pd.to_numeric(df2[args.target_col], errors="coerce").to_numpy(dtype=float)
    g2 = groups_all[mask_h2]
    X_base = df2[base_cols].to_numpy(dtype=float)
    X_full = df2[full_cols].to_numpy(dtype=float)
    splits2 = make_group_shuffle_splits(
        X_full,
        y2,
        g2,
        n_splits=args.h2_splits,
        test_size=args.h2_test_size,
        seed=args.seed + 17,
    )

    ridge_factory = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    fold_base, _ = run_fold_eval(X_base, y2, splits2, ridge_factory)
    fold_full, _ = run_fold_eval(X_full, y2, splits2, ridge_factory)
    fold_base["set"] = "base_no_coord"
    fold_full["set"] = "full_with_coord"
    h2_fold = pd.concat([fold_base, fold_full], ignore_index=True)

    h2_r2_diff = fold_full["r2"].to_numpy(dtype=float) - fold_base["r2"].to_numpy(dtype=float)
    h2_mae_diff = fold_base["mae"].to_numpy(dtype=float) - fold_full["mae"].to_numpy(dtype=float)
    h2_r2_t2 = stats.ttest_1samp(h2_r2_diff, popmean=0.0, nan_policy="omit")
    h2_mae_t2 = stats.ttest_1samp(h2_mae_diff, popmean=0.0, nan_policy="omit")
    h2_r2_t_p = (
        float(h2_r2_t2.pvalue / 2.0) if np.isfinite(h2_r2_t2.pvalue) and float(np.mean(h2_r2_diff)) > 0 else
        (float(1.0 - h2_r2_t2.pvalue / 2.0) if np.isfinite(h2_r2_t2.pvalue) else float("nan"))
    )
    h2_mae_t_p = (
        float(h2_mae_t2.pvalue / 2.0) if np.isfinite(h2_mae_t2.pvalue) and float(np.mean(h2_mae_diff)) > 0 else
        (float(1.0 - h2_mae_t2.pvalue / 2.0) if np.isfinite(h2_mae_t2.pvalue) else float("nan"))
    )
    h2_r2_perm_p = paired_sign_permutation_p(h2_r2_diff, args.perm_iter, args.seed + 2, alternative="greater")
    h2_mae_perm_p = paired_sign_permutation_p(h2_mae_diff, args.perm_iter, args.seed + 3, alternative="greater")

    h2_partial_rows: List[Dict[str, float]] = []
    for i, feat in enumerate(coord_features):
        row = residual_partial_spearman(
            df=df2,
            x_col=feat,
            y_col=args.target_col,
            conf_cols=confounders,
            n_boot=args.bootstrap_iter,
            seed=args.seed + 100 + i,
        )
        h2_partial_rows.append(row)
    h2_partial_df = pd.DataFrame(h2_partial_rows)

    # H3: landing backlean amount should be negatively associated with jump distance.
    # Prefer direction-aligned trunk-lean features when available; fallback to signed/raw.
    df_h3 = df.copy()
    h3_feat_map: List[Tuple[str, str, str]] = []
    if "landing_backlean_deg" in df_h3.columns:
        h3_feat_map.append(("landing_backlean_deg", "landing_trunk_lean_forward_deg", "primary"))
    elif "landing_trunk_lean_forward_deg" in df_h3.columns:
        df_h3["landing_backlean_deg"] = -pd.to_numeric(df_h3["landing_trunk_lean_forward_deg"], errors="coerce")
        h3_feat_map.append(("landing_backlean_deg", "landing_trunk_lean_forward_deg", "primary"))
    elif "landing_trunk_lean_signed_deg" in df_h3.columns:
        df_h3["landing_backlean_deg"] = -pd.to_numeric(df_h3["landing_trunk_lean_signed_deg"], errors="coerce")
        h3_feat_map.append(("landing_backlean_deg", "landing_trunk_lean_signed_deg", "primary"))
    elif "landing_trunk_lean_deg" in df_h3.columns:
        df_h3["landing_backlean_deg"] = -pd.to_numeric(df_h3["landing_trunk_lean_deg"], errors="coerce")
        h3_feat_map.append(("landing_backlean_deg", "landing_trunk_lean_deg", "primary_fallback"))

    if "landing_mid_backlean_deg" in df_h3.columns:
        h3_feat_map.append(("landing_mid_backlean_deg", "landing_mid_trunk_lean_forward_deg", "secondary"))
    elif "landing_mid_trunk_lean_forward_deg" in df_h3.columns:
        df_h3["landing_mid_backlean_deg"] = -pd.to_numeric(df_h3["landing_mid_trunk_lean_forward_deg"], errors="coerce")
        h3_feat_map.append(("landing_mid_backlean_deg", "landing_mid_trunk_lean_forward_deg", "secondary"))
    elif "landing_mid_trunk_lean_signed_deg" in df_h3.columns:
        df_h3["landing_mid_backlean_deg"] = -pd.to_numeric(df_h3["landing_mid_trunk_lean_signed_deg"], errors="coerce")
        h3_feat_map.append(("landing_mid_backlean_deg", "landing_mid_trunk_lean_signed_deg", "secondary"))
    elif "landing_mid_trunk_lean_deg" in df_h3.columns:
        df_h3["landing_mid_backlean_deg"] = -pd.to_numeric(df_h3["landing_mid_trunk_lean_deg"], errors="coerce")
        h3_feat_map.append(("landing_mid_backlean_deg", "landing_mid_trunk_lean_deg", "secondary_fallback"))

    if "trunk_lean_forward_takeoff_to_landing" in df_h3.columns:
        df_h3["backlean_change_takeoff_to_landing"] = -pd.to_numeric(
            df_h3["trunk_lean_forward_takeoff_to_landing"], errors="coerce"
        )
        h3_feat_map.append(
            ("backlean_change_takeoff_to_landing", "trunk_lean_forward_takeoff_to_landing", "secondary")
        )
    elif "trunk_lean_signed_takeoff_to_landing" in df_h3.columns:
        df_h3["backlean_change_takeoff_to_landing"] = -pd.to_numeric(
            df_h3["trunk_lean_signed_takeoff_to_landing"], errors="coerce"
        )
        h3_feat_map.append(
            ("backlean_change_takeoff_to_landing", "trunk_lean_signed_takeoff_to_landing", "secondary_fallback")
        )

    h3_rows: List[Dict[str, float]] = []
    for i, (feat, raw_feat, role) in enumerate(h3_feat_map):
        row = residual_partial_spearman(
            df=df_h3,
            x_col=feat,
            y_col=args.target_col,
            conf_cols=confounders,
            n_boot=args.bootstrap_iter,
            seed=args.seed + 200 + i,
        )
        if np.isfinite(row.get("rho", float("nan"))):
            rho = float(row["rho"])
            p_two = float(row["p_two_sided"])
            p_one_neg = p_two / 2.0 if rho < 0 else 1.0 - p_two / 2.0
        else:
            p_one_neg = float("nan")
        row["raw_feature"] = raw_feat
        row["definition"] = f"{feat} = -{raw_feat}"
        row["role"] = role
        row["p_one_sided_negative"] = p_one_neg
        row["direction"] = "negative" if (np.isfinite(row.get("rho", float("nan"))) and float(row["rho"]) < 0) else "positive"
        h3_rows.append(row)
    h3_df = pd.DataFrame(h3_rows)
    if len(h3_df) > 0:
        h3_df["p_two_sided_holm"] = holm_adjust_pvalues(pd.to_numeric(h3_df["p_two_sided"], errors="coerce").to_numpy(dtype=float))
        h3_df["is_significant_holm"] = (
            pd.to_numeric(h3_df["p_two_sided_holm"], errors="coerce") < 0.05
        ).astype(bool)
        h3_supported = bool(h3_df["is_significant_holm"].any())
    else:
        h3_df["p_two_sided_holm"] = []
        h3_df["is_significant_holm"] = []
        h3_supported = False

    # Save tables
    h1_fold_path = out_dir / "h1_fold_metrics.csv"
    h2_fold_path = out_dir / "h2_fold_metrics.csv"
    h2_partial_path = out_dir / "h2_partial_corr_coordination.csv"
    h3_path = out_dir / "h3_partial_corr_landing.csv"
    h_summary_path = out_dir / "hypothesis_summary.csv"
    h_json_path = out_dir / "hypothesis_summary.json"
    h1_oof_path = out_dir / "h1_oof_compare.csv"

    h1_fold.to_csv(h1_fold_path, index=False, encoding="utf-8")
    h2_fold.to_csv(h2_fold_path, index=False, encoding="utf-8")
    h2_partial_df.to_csv(h2_partial_path, index=False, encoding="utf-8")
    h3_df.to_csv(h3_path, index=False, encoding="utf-8")

    h1_oof_df = pd.DataFrame(pred_rows)
    h1_oof_df["abs_err_phase"] = np.abs(h1_oof_df["y_pred_phase"] - h1_oof_df["y_true"])
    h1_oof_df["abs_err_non_phase"] = np.abs(h1_oof_df["y_pred_non_phase"] - h1_oof_df["y_true"])
    h1_oof_df["abs_err_gain_phase"] = h1_oof_df["abs_err_non_phase"] - h1_oof_df["abs_err_phase"]
    h1_oof_df.to_csv(h1_oof_path, index=False, encoding="utf-8")

    h_rows = [
        {
            "hypothesis": "H1_phase_alignment_beats_non_phase",
            "test": f"RF GroupShuffleSplit paired (kinematic-only), n={len(fold_phase)}",
            "delta_mae_mean": float(np.mean(mae_diff)),
            "delta_mae_ci_low": mae_ci[0],
            "delta_mae_ci_high": mae_ci[1],
            "delta_mae_ttest_p": mae_t_p,
            "delta_mae_perm_p": mae_perm_p,
            "delta_r2_mean": float(np.mean(r2_diff)),
            "delta_r2_ci_low": r2_ci[0],
            "delta_r2_ci_high": r2_ci[1],
            "delta_r2_ttest_p": r2_t_p,
            "delta_r2_perm_p": r2_perm_p,
            "supported": bool(
                float(np.mean(mae_diff)) > 0
                and float(np.mean(r2_diff)) > 0
                and mae_perm_p < 0.05
                and r2_perm_p < 0.05
            ),
        },
        {
            "hypothesis": "H2_coordination_block_adds_independent_value",
            "test": f"Ridge GroupShuffleSplit paired, n={len(fold_full)}",
            "delta_mae_mean": float(np.mean(h2_mae_diff)),
            "delta_mae_ci_low": bootstrap_ci_mean(h2_mae_diff, args.bootstrap_iter, args.seed + 4)[0],
            "delta_mae_ci_high": bootstrap_ci_mean(h2_mae_diff, args.bootstrap_iter, args.seed + 4)[1],
            "delta_mae_ttest_p": h2_mae_t_p,
            "delta_mae_perm_p": h2_mae_perm_p,
            "delta_r2_mean": float(np.mean(h2_r2_diff)),
            "delta_r2_ci_low": bootstrap_ci_mean(h2_r2_diff, args.bootstrap_iter, args.seed + 5)[0],
            "delta_r2_ci_high": bootstrap_ci_mean(h2_r2_diff, args.bootstrap_iter, args.seed + 5)[1],
            "delta_r2_ttest_p": h2_r2_t_p,
            "delta_r2_perm_p": h2_r2_perm_p,
            "supported": bool(float(np.mean(h2_r2_diff)) > 0 and h2_r2_perm_p < 0.05),
        },
        {
            "hypothesis": "H3_landing_trunk_control_association",
            "test": "partial Spearman on landing trunk-control block (two-sided, Holm-corrected)",
            "delta_mae_mean": float("nan"),
            "delta_mae_ci_low": float("nan"),
            "delta_mae_ci_high": float("nan"),
            "delta_mae_ttest_p": float("nan"),
            "delta_mae_perm_p": float("nan"),
            "delta_r2_mean": float("nan"),
            "delta_r2_ci_low": float("nan"),
            "delta_r2_ci_high": float("nan"),
            "delta_r2_ttest_p": float("nan"),
            "delta_r2_perm_p": float("nan"),
            "supported": h3_supported,
        },
    ]
    h_summary_df = pd.DataFrame(h_rows)
    h_summary_df.to_csv(h_summary_path, index=False, encoding="utf-8")

    payload = {
        "inputs": {
            "features_csv": args.features_csv,
            "feature_set_json": args.feature_set_json,
            "target_col": args.target_col,
            "group_col": args.group_col,
            "cv_splits": args.cv_splits,
            "h1_splits": args.h1_splits,
            "h1_test_size": args.h1_test_size,
            "h2_splits": args.h2_splits,
            "h2_test_size": args.h2_test_size,
        },
        "h1": h_rows[0],
        "h2": h_rows[1],
        "h3": h_rows[2],
        "h1_phase_kinematic_features": h1_phase_cols,
        "h1_non_phase_coarse_features": h1_non_phase_cols,
        "coord_features": coord_features,
        "h3_candidates": h3_feat_map,
        "h3_best_feature": (
            h3_df.sort_values("p_two_sided_holm", ascending=True).iloc[0].to_dict()
            if len(h3_df) > 0 and "p_two_sided_holm" in h3_df.columns
            else {}
        ),
        "confounders": confounders,
    }
    h_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved: {h1_fold_path}")
    print(f"saved: {h2_fold_path}")
    print(f"saved: {h2_partial_path}")
    print(f"saved: {h3_path}")
    print(f"saved: {h1_oof_path}")
    print(f"saved: {h_summary_path}")
    print(f"saved: {h_json_path}")
    print("\n[Hypothesis summary]")
    print(h_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
