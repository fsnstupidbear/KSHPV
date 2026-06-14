from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

PHASES: Tuple[str, ...] = ("entry", "takeoff", "takeoff_mid", "peak", "landing_mid", "landing")

FIXED_FEATURES = {
    "height_cm",
    "weight_kg",
    "prep_frames",
    "takeoff_to_peak_frames",
    "peak_to_landing_frames",
    "flight_frames",
    "knee_ext_takeoff_to_peak",
    "arm_raise_takeoff_to_peak",
    "trunk_lean_takeoff_to_landing",
}

SUFFIX_FEATURES = (
    "_arm_raise",
    "_leg_forward",
    "_knee_angle",
    "_hip_angle",
    "_trunk_lean_deg",
    "_kps_conf_mean",
    "_kps_valid_ratio",
)


def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in FIXED_FEATURES or c.endswith(SUFFIX_FEATURES):
            cols.append(c)
    return cols


def build_person_uid(df: pd.DataFrame) -> pd.Series:
    if {"prefix", "subject_id"}.issubset(df.columns):
        return df["prefix"].astype(str) + "__" + df["subject_id"].astype(str)
    if "video_stem" in df.columns:
        stems = df["video_stem"].astype(str)
        out = []
        for s in stems:
            parts = s.rsplit("_", 5)
            if len(parts) == 6:
                out.append(parts[0] + "__" + parts[1])
            else:
                out.append(s)
        return pd.Series(out, index=df.index)
    if "video_name" in df.columns:
        return df["video_name"].astype(str)
    return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)


def resolve_groups(
    df: pd.DataFrame,
    group_col: str = "person_uid",
    allow_none: bool = False,
) -> Optional[np.ndarray]:
    key = str(group_col).strip().lower()
    if key in {"", "none", "null", "na", "no"}:
        if allow_none:
            return None
        raise ValueError("Grouping is required, but group_col resolves to none.")
    if key == "person_uid":
        return build_person_uid(df).astype(str).to_numpy()
    if group_col in df.columns:
        return df[group_col].astype(str).to_numpy()
    raise ValueError(f"group_col not found: {group_col}")


def load_recommended_features(
    df: pd.DataFrame,
    feature_set_json: str,
    fallback_k: int = 15,
) -> List[str]:
    if feature_set_json:
        p = Path(feature_set_json)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            feats = data.get("recommended_features", [])
            if isinstance(feats, list) and len(feats) > 0:
                return [f for f in feats if f in df.columns]
    return pick_feature_columns(df)[:fallback_k]


def make_group_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_groups = len(set(groups.tolist()))
    n_splits = min(cv_splits, n_groups)
    if n_splits < 2:
        raise ValueError(f"Not enough groups for GroupKFold: {n_groups}")
    return list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))


def make_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_splits = min(cv_splits, len(y))
    if n_splits < 2:
        raise ValueError(f"Not enough rows for KFold: {len(y)}")
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))


def prepare_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "distance_cm",
    group_col: str = "person_uid",
    allow_none_group: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], pd.DataFrame, List[str]]:
    use_cols = [c for c in feature_cols if c in df.columns]
    if len(use_cols) == 0:
        raise ValueError("No usable feature columns.")

    X_df = df[use_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    groups = resolve_groups(df, group_col=group_col, allow_none=allow_none_group)

    mask = np.isfinite(y)
    for c in use_cols:
        mask &= np.isfinite(X_df[c].to_numpy(dtype=float))
    if groups is not None:
        mask &= pd.Series(groups).notna().to_numpy()

    X = X_df.loc[mask, use_cols].to_numpy(dtype=float)
    y = y[mask]
    g = groups[mask] if groups is not None else None
    df_used = df.loc[mask].copy()
    if g is not None:
        df_used["__group__"] = g
    return X, y, g, df_used, use_cols


def eval_splits(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    out_of_fold: bool = False,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    rows: List[Dict[str, float]] = []
    oof = np.full(len(y), np.nan, dtype=float) if out_of_fold else None

    for fold_id, (tr, va) in enumerate(splits, start=1):
        model = model_factory()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[va])
        mae = float(mean_absolute_error(y[va], pred))
        rmse = float(np.sqrt(mean_squared_error(y[va], pred)))
        r2 = float(r2_score(y[va], pred)) if len(va) > 1 else float("nan")
        bias = float(np.mean(pred - y[va]))
        rows.append(
            {
                "fold": fold_id,
                "n_val": int(len(va)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "bias": bias,
            }
        )
        if oof is not None:
            oof[va] = pred

    return pd.DataFrame(rows), oof


def summarize_fold_df(fold_df: pd.DataFrame) -> Dict[str, float]:
    return {
        "n_folds": int(len(fold_df)),
        "mae_mean": float(fold_df["mae"].mean()),
        "mae_std": float(fold_df["mae"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "rmse_mean": float(fold_df["rmse"].mean()),
        "rmse_std": float(fold_df["rmse"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "r2_mean": float(fold_df["r2"].mean()),
        "r2_std": float(fold_df["r2"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
        "bias_mean": float(fold_df["bias"].mean()),
        "bias_std": float(fold_df["bias"].std(ddof=1)) if len(fold_df) > 1 else 0.0,
    }
