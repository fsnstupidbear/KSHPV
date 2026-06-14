from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run main paper analysis from selected phase features."
    )
    parser.add_argument("--features-csv", required=True, help="Input feature CSV, usually phase_features_clean.csv.")
    parser.add_argument(
        "--feature-set-json",
        default="",
        help="JSON produced by diagnose_groups_and_select_features.py (recommended_feature_set.json).",
    )
    parser.add_argument("--target-col", default="distance_cm")
    parser.add_argument(
        "--group-col",
        default="person_uid",
        help='Grouping key for GroupKFold. Use "person_uid" (default) or a column name.',
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--perm-repeats", type=int, default=1, help="Permutation repeats per feature per validation fold.")
    parser.add_argument("--out-dir", default="log/paper_main", help="Output folder.")
    return parser.parse_args()


def load_feature_set(feature_set_json: str) -> List[str]:
    if not feature_set_json:
        return []
    p = Path(feature_set_json)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    feats = data.get("recommended_features", [])
    if not isinstance(feats, list):
        return []
    return [str(x) for x in feats]


def default_feature_pool(df: pd.DataFrame) -> List[str]:
    fixed_cols = {
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
    suffix_cols = (
        "_arm_raise",
        "_leg_forward",
        "_knee_angle",
        "_hip_angle",
        "_trunk_lean_deg",
        "_kps_conf_mean",
        "_kps_valid_ratio",
    )
    out: List[str] = []
    for c in df.columns:
        if c in fixed_cols or c.endswith(suffix_cols):
            out.append(c)
    return out


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


def resolve_groups(df: pd.DataFrame, group_col: str) -> np.ndarray:
    key = str(group_col).strip().lower()
    if key in {"", "none", "null", "na", "no"}:
        raise ValueError("run_paper_main_analysis.py requires grouping; use --group-col person_uid")
    if key == "person_uid":
        return build_person_uid(df).astype(str).to_numpy()
    if group_col in df.columns:
        return df[group_col].astype(str).to_numpy()
    raise ValueError(f"group_col not found: {group_col}")


def make_dataset(
    df: pd.DataFrame,
    features: Sequence[str],
    target_col: str,
    group_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    use = [f for f in features if f in df.columns]
    if not use:
        raise ValueError("No valid features present in dataframe.")
    X_df = df[use].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    g = resolve_groups(df, group_col)

    mask = np.isfinite(y)
    for c in X_df.columns:
        mask &= np.isfinite(X_df[c].to_numpy(dtype=float))

    X = X_df[mask].to_numpy(dtype=float)
    y = y[mask]
    g = g[mask]
    df_used = df.loc[mask].copy()
    df_used["__group__"] = g
    return X, y, g, df_used


def make_group_splits(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    group_n = len(set(groups.tolist()))
    splits = min(n_splits, group_n)
    if splits < 2:
        raise ValueError(f"Not enough groups for GroupKFold: {group_n}")
    return list(GroupKFold(n_splits=splits).split(X, y, groups=groups))


def make_kfold_splits(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = min(n_splits, len(y))
    if splits < 2:
        raise ValueError(f"Not enough rows for KFold: {len(y)}")
    return list(KFold(n_splits=splits, shuffle=True, random_state=seed).split(X, y))


def run_cv(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    df_used: pd.DataFrame,
    out_of_fold: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []

    for fold_id, (tr, va) in enumerate(splits, start=1):
        model = model_factory()
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)

        mae = float(mean_absolute_error(y_va, pred))
        rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
        r2 = float(r2_score(y_va, pred)) if len(y_va) > 1 else float("nan")
        bias = float(np.mean(pred - y_va))
        fold_rows.append(
            {
                "fold": fold_id,
                "n_val": int(len(y_va)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "bias": bias,
            }
        )

        if out_of_fold:
            meta = df_used.iloc[va]
            for i in range(len(va)):
                row = {
                    "fold": fold_id,
                    "y_true": float(y_va[i]),
                    "y_pred": float(pred[i]),
                    "error": float(pred[i] - y_va[i]),
                }
                for col in ["video_name", "video_stem", "gender", "prefix", "__group__"]:
                    if col in meta.columns:
                        row[col] = meta.iloc[i][col]
                oof_rows.append(row)

    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def summarize_cv(fold_df: pd.DataFrame) -> Dict[str, float]:
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


def permutation_importance_on_group_cv(
    model_factory: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    feature_names: Sequence[str],
    seed: int,
    n_repeats: int = 1,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n_repeats = max(int(n_repeats), 1)
    gains = np.zeros((len(splits), n_repeats, X.shape[1]), dtype=float)

    for fold_id, (tr, va) in enumerate(splits):
        model = model_factory()
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]
        model.fit(X_tr, y_tr)
        base_pred = model.predict(X_va)
        base_mae = float(mean_absolute_error(y_va, base_pred))
        for repeat_id in range(n_repeats):
            for j in range(X.shape[1]):
                X_perm = X_va.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                pred = model.predict(X_perm)
                mae = float(mean_absolute_error(y_va, pred))
                gains[fold_id, repeat_id, j] = mae - base_mae

    flat_gains = gains.reshape(-1, X.shape[1])

    out = pd.DataFrame(
        {
            "feature": list(feature_names),
            "perm_mae_gain_mean": flat_gains.mean(axis=0),
            "perm_mae_gain_std": flat_gains.std(axis=0, ddof=1) if flat_gains.shape[0] > 1 else np.zeros(X.shape[1]),
            "perm_repeats_per_fold": n_repeats,
            "perm_total_estimates": flat_gains.shape[0],
        }
    ).sort_values("perm_mae_gain_mean", ascending=False)
    return out.reset_index(drop=True)


def feature_direction(df_used: pd.DataFrame, features: Sequence[str], target_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    y = pd.to_numeric(df_used[target_col], errors="coerce")
    for f in features:
        if f not in df_used.columns:
            continue
        x = pd.to_numeric(df_used[f], errors="coerce")
        corr = x.corr(y, method="spearman")
        rows.append(
            {
                "feature": f,
                "spearman_corr": float(corr) if pd.notna(corr) else float("nan"),
                "direction": (
                    "positive" if pd.notna(corr) and corr > 0 else ("negative" if pd.notna(corr) and corr < 0 else "flat")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_corr", ascending=False).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features_csv)
    rec_features = load_feature_set(args.feature_set_json)
    if not rec_features:
        # Fallback to first 10 in default feature pool.
        rec_features = default_feature_pool(df)[:10]

    feature_sets: Dict[str, List[str]] = {
        "anthro_only": [c for c in ["height_cm", "weight_kg"] if c in df.columns],
        "timing_only": [c for c in ["prep_frames", "takeoff_to_peak_frames", "peak_to_landing_frames", "flight_frames"] if c in df.columns],
        "recommended": [c for c in rec_features if c in df.columns],
    }
    feature_sets["recommended_no_anthro"] = [c for c in feature_sets["recommended"] if c not in {"height_cm", "weight_kg"}]

    # Remove empty sets.
    feature_sets = {k: v for k, v in feature_sets.items() if len(v) > 0}
    if "recommended" not in feature_sets:
        raise SystemExit("No recommended feature set available.")

    summary_rows: List[Dict[str, object]] = []
    cv_fold_rows: List[Dict[str, object]] = []
    all_oof_rows: List[Dict[str, object]] = []

    main_X = main_y = main_groups = None
    main_df_used = None
    main_features: List[str] = []

    for set_name, features in feature_sets.items():
        X, y, groups, df_used = make_dataset(df, features, args.target_col, args.group_col)
        gk_splits = make_group_splits(X, y, groups, args.cv_splits)
        kf_splits = make_kfold_splits(X, y, args.cv_splits, args.seed)

        models: List[Tuple[str, Callable[[], object]]] = [
            ("Ridge", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
            (
                "RandomForest",
                lambda: RandomForestRegressor(
                    n_estimators=args.rf_n_estimators, random_state=args.seed, n_jobs=-1
                ),
            ),
        ]

        for model_name, model_factory in models:
            for cv_name, splits in [("GroupKFold", gk_splits), ("KFold", kf_splits)]:
                need_oof = set_name == "recommended" and model_name == "RandomForest" and cv_name == "GroupKFold"
                fold_df, oof_df = run_cv(
                    model_factory=model_factory,
                    X=X,
                    y=y,
                    splits=splits,
                    df_used=df_used,
                    out_of_fold=need_oof,
                )
                stats = summarize_cv(fold_df)
                summary_rows.append(
                    {
                        "feature_set": set_name,
                        "model": model_name,
                        "cv": cv_name,
                        "n_features": len(features),
                        "usable_rows": len(y),
                        **stats,
                    }
                )
                for _, fr in fold_df.iterrows():
                    cv_fold_rows.append(
                        {
                            "feature_set": set_name,
                            "model": model_name,
                            "cv": cv_name,
                            **fr.to_dict(),
                        }
                    )
                if need_oof and len(oof_df) > 0:
                    oof_df["feature_set"] = set_name
                    oof_df["model"] = model_name
                    oof_df["cv"] = cv_name
                    all_oof_rows.extend(oof_df.to_dict(orient="records"))

                    main_X, main_y, main_groups = X, y, groups
                    main_df_used = df_used
                    main_features = list(features)

    summary_df = pd.DataFrame(summary_rows).sort_values(["cv", "mae_mean", "feature_set", "model"])
    fold_df = pd.DataFrame(cv_fold_rows)
    oof_df = pd.DataFrame(all_oof_rows)

    if main_X is None or main_df_used is None:
        raise SystemExit("Main model OOF results missing; check data and splits.")

    # Group-level error summary from main model OOF predictions.
    group_err_df = (
        oof_df.groupby("__group__", as_index=False)
        .agg(
            n=("error", "size"),
            mae=("error", lambda s: float(np.mean(np.abs(s)))),
            rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            bias=("error", "mean"),
        )
        .rename(columns={"__group__": "group"})
        .sort_values("mae", ascending=False)
        .reset_index(drop=True)
    )

    # Main-model importance and direction.
    main_model_factory = lambda: RandomForestRegressor(
        n_estimators=args.rf_n_estimators, random_state=args.seed, n_jobs=-1
    )
    main_gk_splits = make_group_splits(main_X, main_y, main_groups, args.cv_splits)
    imp_df = permutation_importance_on_group_cv(
        model_factory=main_model_factory,
        X=main_X,
        y=main_y,
        splits=main_gk_splits,
        feature_names=main_features,
        seed=args.seed,
        n_repeats=args.perm_repeats,
    )
    dir_df = feature_direction(main_df_used, main_features, args.target_col)

    # Save outputs.
    summary_path = out_dir / "cv_summary.csv"
    folds_path = out_dir / "cv_folds.csv"
    oof_path = out_dir / "oof_predictions_main.csv"
    group_err_path = out_dir / "group_error_main.csv"
    imp_path = out_dir / "feature_importance_main.csv"
    dir_path = out_dir / "feature_direction_main.csv"
    json_path = out_dir / "paper_main_summary.json"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    fold_df.to_csv(folds_path, index=False, encoding="utf-8")
    oof_df.to_csv(oof_path, index=False, encoding="utf-8")
    group_err_df.to_csv(group_err_path, index=False, encoding="utf-8")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8")
    dir_df.to_csv(dir_path, index=False, encoding="utf-8")

    best_group_rf = summary_df[
        (summary_df["feature_set"] == "recommended")
        & (summary_df["model"] == "RandomForest")
        & (summary_df["cv"] == "GroupKFold")
    ].iloc[0]
    best_kfold_rf = summary_df[
        (summary_df["feature_set"] == "recommended")
        & (summary_df["model"] == "RandomForest")
        & (summary_df["cv"] == "KFold")
    ].iloc[0]

    payload = {
        "recommended_features": main_features,
        "usable_rows": int(len(main_y)),
        "group_count": int(len(set(main_groups.tolist()))),
        "groupkfold_rf": {
            "mae_mean": float(best_group_rf["mae_mean"]),
            "mae_std": float(best_group_rf["mae_std"]),
            "r2_mean": float(best_group_rf["r2_mean"]),
            "r2_std": float(best_group_rf["r2_std"]),
        },
        "kfold_rf": {
            "mae_mean": float(best_kfold_rf["mae_mean"]),
            "mae_std": float(best_kfold_rf["mae_std"]),
            "r2_mean": float(best_kfold_rf["r2_mean"]),
            "r2_std": float(best_kfold_rf["r2_std"]),
        },
        "worst_groups_by_mae": group_err_df.head(5).to_dict(orient="records"),
        "top_importances": imp_df.head(10).to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"usable_rows={len(main_y)} groups={len(set(main_groups.tolist()))}")
    print(f"saved: {summary_path}")
    print(f"saved: {folds_path}")
    print(f"saved: {oof_path}")
    print(f"saved: {group_err_path}")
    print(f"saved: {imp_path}")
    print(f"saved: {dir_path}")
    print(f"saved: {json_path}")

    print("\n[Recommended set RF]")
    print(
        summary_df[
            (summary_df["feature_set"] == "recommended")
            & (summary_df["model"] == "RandomForest")
        ][["cv", "mae_mean", "mae_std", "r2_mean", "r2_std"]].to_string(index=False)
    )

    print("\n[Top 8 importance by MAE gain]")
    print(imp_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
