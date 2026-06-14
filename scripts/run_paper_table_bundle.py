from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_publication_style() -> None:
    # Use a print-friendly white theme to avoid gray bands/line artifacts in Word.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.9,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.55,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.9)


def style_legend(legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    frame.set_edgecolor("black")
    frame.set_linewidth(0.8)
    legend.set_zorder(20)


def humanize_feature_label(name: str) -> str:
    mapping = {
        "height_cm": "Height (cm)",
        "weight_kg": "Weight (kg)",
        "flight_frames": "Flight frames",
        "peak_to_landing_frames": "Peak to landing frames",
        "peak_arm_raise": "Peak arm raise",
        "landing_hip_angle": "Landing hip angle",
        "landing_knee_angle": "Landing knee angle",
        "landing_mid_kps_conf_mean": "Landing-mid keypoint confidence",
        "takeoff_kps_conf_mean": "Takeoff keypoint confidence",
        "takeoff_mid_kps_conf_mean": "Takeoff-mid keypoint confidence",
        "takeoff_mid_arm_raise": "Takeoff-mid arm raise",
        "takeoff_mid_leg_forward": "Takeoff-mid leg forward",
        "takeoff_mid_hip_angle": "Takeoff-mid hip angle",
        "takeoff_mid_knee_angle": "Takeoff-mid knee angle",
        "takeoff_trunk_lean_deg": "Takeoff trunk lean (deg)",
        "takeoff_mid_trunk_lean_deg": "Takeoff-mid trunk lean (deg)",
        "knee_ext_takeoff_to_peak": "Knee extension: takeoff to peak",
        "trunk_lean_takeoff_to_landing": "Trunk lean: takeoff to landing",
    }
    if name in mapping:
        return mapping[name]
    label = str(name).replace("_", " ").replace(":", " - ")
    return label.capitalize()


def humanize_subgroup_label(subgroup_col: str, subgroup_level: str) -> str:
    subgroup_col = str(subgroup_col)
    subgroup_level = str(subgroup_level)
    if subgroup_col == "gender":
        gender_map = {"M": "Male", "F": "Female"}
        return gender_map.get(subgroup_level, subgroup_level)
    if subgroup_col == "bmi_group":
        bmi_map = {
            "under": "Underweight",
            "normal": "Normal BMI",
            "over": "Overweight",
            "obese": "Obese",
        }
        return bmi_map.get(subgroup_level, subgroup_level.replace("_", " ").title())
    return f"{subgroup_col.replace('_', ' ').title()}: {subgroup_level.replace('_', ' ').title()}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export core paper tables/figures.")
    parser.add_argument("--raw-features-csv", default="log/phase_features.csv")
    parser.add_argument("--clean-features-csv", default="log/phase_features_clean_filtered_v2.csv")
    parser.add_argument("--feature-set-json", default="log/feature_diag_person_uid/recommended_feature_set.json")
    parser.add_argument("--results-csv", default="log/results.csv")
    parser.add_argument("--target-col", default="distance_cm")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-iter", type=int, default=800)
    parser.add_argument("--perm-iter", type=int, default=2000)
    parser.add_argument("--noise-repeats", type=int, default=4)
    parser.add_argument("--noise-levels", default="0,0.02,0.05,0.10")
    parser.add_argument("--missing-rates", default="0,0.05,0.10,0.20")
    parser.add_argument("--qc-thresholds", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45")
    parser.add_argument("--out-dir", default="log/paper_assets_core")
    parser.add_argument("--intermediate-dir", default="")
    parser.add_argument("--figure-format", default="png", choices=["png", "pdf"])
    parser.add_argument("--figure-dpi", type=int, default=260)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_person_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "person_uid" in out.columns:
        return out
    if {"prefix", "subject_id"}.issubset(out.columns):
        out["person_uid"] = out["prefix"].astype(str) + "__" + out["subject_id"].astype(str)
        return out
    if "video_stem" in out.columns:
        vals = []
        for s in out["video_stem"].astype(str):
            parts = s.rsplit("_", 5)
            vals.append(parts[0] + "__" + parts[1] if len(parts) == 6 else s)
        out["person_uid"] = vals
        return out
    out["person_uid"] = [f"row_{i}" for i in range(len(out))]
    return out


def copy_csv(src: Path, dst: Path) -> pd.DataFrame:
    if not src.exists():
        raise FileNotFoundError(f"Missing required csv: {src}")
    shutil.copyfile(src, dst)
    return pd.read_csv(dst)


def save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_table_01(results_csv: Path, raw_csv: Path, clean_csv: Path, target_col: str) -> pd.DataFrame:
    n_results = int(pd.read_csv(results_csv).shape[0]) if results_csv.exists() else np.nan
    raw_df = ensure_person_uid(pd.read_csv(raw_csv))
    clean_df = ensure_person_uid(pd.read_csv(clean_csv))

    rows: List[Dict[str, object]] = [
        {"metric": "raw_video_rows", "value": n_results, "note": "rows in results.csv"},
        {"metric": "feature_rows", "value": int(len(raw_df)), "note": "rows in phase_features.csv"},
        {
            "metric": "quality_pass_rows",
            "value": int(raw_df["quality_pass"].sum()) if "quality_pass" in raw_df.columns else np.nan,
            "note": "rows after quality gate",
        },
        {"metric": "clean_rows", "value": int(len(clean_df)), "note": "rows after outlier filtering"},
        {
            "metric": "clean_ratio_vs_feature",
            "value": float(len(clean_df) / max(len(raw_df), 1)),
            "note": "clean/feature",
        },
        {"metric": "subject_count_clean", "value": int(clean_df["person_uid"].nunique()), "note": "unique person_uid"},
    ]

    if "gender" in clean_df.columns:
        vc = clean_df["gender"].astype(str).value_counts()
        rows.append({"metric": "gender_M_rows", "value": int(vc.get("M", 0)), "note": "clean rows"})
        rows.append({"metric": "gender_F_rows", "value": int(vc.get("F", 0)), "note": "clean rows"})
    if "height_cm" in clean_df.columns:
        h = pd.to_numeric(clean_df["height_cm"], errors="coerce")
        rows.append({"metric": "height_cm_mean", "value": float(h.mean()), "note": "clean"})
        rows.append({"metric": "height_cm_std", "value": float(h.std()), "note": "clean"})
    if "weight_kg" in clean_df.columns:
        w = pd.to_numeric(clean_df["weight_kg"], errors="coerce")
        rows.append({"metric": "weight_kg_mean", "value": float(w.mean()), "note": "clean"})
        rows.append({"metric": "weight_kg_std", "value": float(w.std()), "note": "clean"})
    if target_col in clean_df.columns:
        y = pd.to_numeric(clean_df[target_col], errors="coerce")
        rows.append({"metric": "distance_cm_min", "value": float(y.min()), "note": "clean"})
        rows.append({"metric": "distance_cm_max", "value": float(y.max()), "note": "clean"})
        rows.append({"metric": "distance_cm_mean", "value": float(y.mean()), "note": "clean"})
        rows.append({"metric": "distance_cm_std", "value": float(y.std()), "note": "clean"})
    return pd.DataFrame(rows)


def fig_main_performance(cv_csv: Path, out_fig: Path, out_data: Path, dpi: int) -> None:
    df = pd.read_csv(cv_csv)
    sub = df[(df["cv"] == "GroupKFold") & (df["model"] == "RandomForest")].copy()
    if sub.empty:
        raise ValueError("No RandomForest GroupKFold rows in cv_summary.csv")
    sub = sub.sort_values("r2_mean", ascending=False).reset_index(drop=True)
    sub.to_csv(out_data, index=False, encoding="utf-8")
    label_map = {
        "recommended": "Phase + anthropometrics",
        "recommended_no_anthro": "Phase only",
        "anthro_only": "Anthropometrics only",
        "timing_only": "Timing only",
    }
    labels = [label_map.get(x, str(x)) for x in sub["feature_set"].astype(str)]
    x = np.arange(len(sub))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    style_axis(ax)
    ax.bar(
        x - w / 2,
        sub["mae_mean"],
        yerr=sub["mae_std"],
        width=w,
        label="MAE (cm)",
        color="#4E79A7",
        ecolor="#222222",
        capsize=3,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE (cm)")
    mae_max = float((pd.to_numeric(sub["mae_mean"], errors="coerce") + pd.to_numeric(sub["mae_std"], errors="coerce")).max())
    ax.set_ylim(0, mae_max * 1.35)
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    style_axis(ax2)
    ax2.bar(
        x + w / 2,
        sub["r2_mean"],
        yerr=sub["r2_std"],
        width=w,
        label="R2",
        color="#59A14F",
        ecolor="#222222",
        capsize=3,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax2.set_ylabel(r"$R^2$")
    r2_max = float((pd.to_numeric(sub["r2_mean"], errors="coerce") + pd.to_numeric(sub["r2_std"], errors="coerce")).max())
    ax2.set_ylim(0, r2_max * 1.35)

    # Explicit mean +/- SD labels address reviewer requests for reported variability.
    for xi, mean, sd in zip(x - w / 2, sub["mae_mean"], sub["mae_std"]):
        ax.text(
            xi,
            float(mean) + float(sd) + mae_max * 0.025,
            f"{float(mean):.2f}\n$\\pm$ {float(sd):.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#1F4E79",
            zorder=5,
        )
    for xi, mean, sd in zip(x + w / 2, sub["r2_mean"], sub["r2_std"]):
        ax2.text(
            xi,
            float(mean) + float(sd) + r2_max * 0.025,
            f"{float(mean):.3f}\n$\\pm$ {float(sd):.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#2F6B2F",
            zorder=5,
        )

    ax.set_title("Main model comparison (RF, GroupKFold)")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    leg = ax.legend(
        l1 + l2,
        lb1 + lb2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        borderaxespad=0.2,
    )
    style_legend(leg)
    save_fig(fig, out_fig, dpi)


def fig_qc_tradeoff(qc_csv: Path, out_fig: Path, out_data: Path, dpi: int) -> None:
    df = pd.read_csv(qc_csv)
    gate = df[df["condition"] == "qc_gate"].copy()
    gate["min_phase_conf_thr"] = pd.to_numeric(gate["min_phase_conf_thr"], errors="coerce")
    gate = gate.sort_values("min_phase_conf_thr").reset_index(drop=True)
    gate.to_csv(out_data, index=False, encoding="utf-8")
    no_qc = df[df["condition"] == "no_qc_gate"]
    no_qc_mae = float(no_qc.iloc[0]["mae_mean"]) if len(no_qc) > 0 else np.nan
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    style_axis(ax)
    ax.plot(gate["min_phase_conf_thr"], gate["mae_mean"], marker="o", lw=2, label="QC MAE")
    if np.isfinite(no_qc_mae):
        ax.axhline(no_qc_mae, ls="--", color="tab:red", label="No-QC MAE")
    ax.set_xlabel("min_phase_conf threshold")
    ax.set_ylabel("MAE (cm)")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    style_axis(ax2)
    ax2.plot(gate["min_phase_conf_thr"], gate["keep_ratio"], marker="s", lw=2, color="tab:green", label="Keep ratio")
    ax2.set_ylabel("Keep ratio")
    ax.set_title("QC threshold tradeoff")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    leg = ax.legend(l1 + l2, lb1 + lb2, loc="upper right", frameon=True)
    style_legend(leg)
    save_fig(fig, out_fig, dpi)


def fig_importance_positive(imp_csv: Path, out_fig: Path, out_data: Path, dpi: int, topk: int) -> None:
    df = pd.read_csv(imp_csv)
    df["perm_mae_gain_mean"] = pd.to_numeric(df["perm_mae_gain_mean"], errors="coerce")
    df["perm_mae_gain_std"] = pd.to_numeric(df["perm_mae_gain_std"], errors="coerce")
    df = df[df["perm_mae_gain_mean"] > 0].copy()
    if df.empty:
        raise ValueError("No positive importance features found.")
    df = df.sort_values("perm_mae_gain_mean", ascending=False).head(topk)
    df = df.sort_values("perm_mae_gain_mean", ascending=True).reset_index(drop=True)
    df["feature_label"] = df["feature"].map(humanize_feature_label)
    df.to_csv(out_data, index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9.2, max(4.4, 0.36 * len(df))))
    style_axis(ax)
    ax.barh(
        df["feature_label"],
        df["perm_mae_gain_mean"],
        xerr=df["perm_mae_gain_std"],
        color="#F28E2B",
        ecolor="black",
        capsize=2,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_xlabel("Permutation MAE gain")
    ax.set_title(f"Top-{len(df)} positive feature importance")
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, out_fig, dpi)


def fig_robust_gender_bmi(sub_csv: Path, out_fig: Path, out_data: Path, dpi: int) -> None:
    df = pd.read_csv(sub_csv)
    sub = df[df["subgroup_col"].isin(["gender", "bmi_group"])].copy()
    if sub.empty:
        sub = df.copy()
    sub = sub.sort_values(["subgroup_col", "mae"], ascending=[True, False]).reset_index(drop=True)
    sub["label"] = [
        humanize_subgroup_label(col, level)
        for col, level in zip(sub["subgroup_col"].astype(str), sub["subgroup_level"].astype(str))
    ]
    sub.to_csv(out_data, index=False, encoding="utf-8")
    mae = pd.to_numeric(sub["mae"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(sub["mae_ci_low"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(sub["mae_ci_high"], errors="coerce").to_numpy(dtype=float)
    err_low = np.where(np.isfinite(low), np.maximum(mae - low, 0.0), 0.0)
    err_high = np.where(np.isfinite(high), np.maximum(high - mae, 0.0), 0.0)
    fig, ax = plt.subplots(figsize=(9.0, max(4.3, 0.42 * len(sub))))
    style_axis(ax)
    y = np.arange(len(sub))
    ax.barh(
        y,
        mae,
        xerr=np.vstack([err_low, err_high]),
        color="#76B7B2",
        ecolor="black",
        capsize=2,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("MAE (cm)")
    ax.set_title("Robustness across gender/BMI")
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, out_fig, dpi)


def fig_noise_missing(noise_csv: Path, out_fig: Path, out_data: Path, dpi: int) -> None:
    df = pd.read_csv(noise_csv)
    df = df.sort_values(["noise_level", "missing_rate"]).reset_index(drop=True)
    df.to_csv(out_data, index=False, encoding="utf-8")
    piv = df.pivot(index="noise_level", columns="missing_rate", values="mae_mean")
    xvals = list(piv.columns)
    yvals = list(piv.index)
    z = piv.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    style_axis(ax)
    im = ax.imshow(z, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(xvals)))
    ax.set_xticklabels([f"{x:.2f}" for x in xvals])
    ax.set_yticks(np.arange(len(yvals)))
    ax.set_yticklabels([f"{y:.2f}" for y in yvals])
    ax.set_xlabel("Missing rate")
    ax.set_ylabel("Noise level")
    ax.set_title("Noise/missingness sensitivity (MAE)")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("MAE (cm)")
    save_fig(fig, out_fig, dpi)


def run_analyses(root: Path, work: Path, args: argparse.Namespace) -> None:
    py = sys.executable
    run_cmd(
        [
            py,
            "pose_recognization/run_paper_main_analysis.py",
            "--features-csv",
            args.clean_features_csv,
            "--feature-set-json",
            args.feature_set_json,
            "--out-dir",
            str(work / "paper_main"),
            "--cv-splits",
            str(args.cv_splits),
            "--seed",
            str(args.seed),
            "--rf-n-estimators",
            str(args.rf_n_estimators),
        ],
        cwd=root,
    )
    run_cmd(
        [
            py,
            "pose_recognization/run_hypothesis_tests.py",
            "--features-csv",
            args.clean_features_csv,
            "--feature-set-json",
            args.feature_set_json,
            "--out-dir",
            str(work / "hypothesis"),
            "--cv-splits",
            str(args.cv_splits),
            "--seed",
            str(args.seed),
            "--rf-n-estimators",
            str(args.rf_n_estimators),
            "--bootstrap-iter",
            str(args.bootstrap_iter),
            "--perm-iter",
            str(args.perm_iter),
        ],
        cwd=root,
    )
    run_cmd(
        [
            py,
            "pose_recognization/run_qc_sensitivity.py",
            "--features-csv",
            args.raw_features_csv,
            "--feature-set-json",
            args.feature_set_json,
            "--out-dir",
            str(work / "qc"),
            "--thresholds",
            args.qc_thresholds,
            "--cv-splits",
            str(args.cv_splits),
            "--seed",
            str(args.seed),
            "--rf-n-estimators",
            str(args.rf_n_estimators),
        ],
        cwd=root,
    )
    run_cmd(
        [
            py,
            "pose_recognization/run_robustness_subgroups.py",
            "--features-csv",
            args.clean_features_csv,
            "--feature-set-json",
            args.feature_set_json,
            "--out-dir",
            str(work / "robustness"),
            "--cv-splits",
            str(args.cv_splits),
            "--seed",
            str(args.seed),
            "--rf-n-estimators",
            str(args.rf_n_estimators),
            "--bootstrap-iter",
            str(args.bootstrap_iter),
        ],
        cwd=root,
    )
    run_cmd(
        [
            py,
            "pose_recognization/run_noise_missing_sensitivity.py",
            "--features-csv",
            args.clean_features_csv,
            "--feature-set-json",
            args.feature_set_json,
            "--out-dir",
            str(work / "noise"),
            "--cv-splits",
            str(args.cv_splits),
            "--seed",
            str(args.seed),
            "--rf-n-estimators",
            str(args.rf_n_estimators),
            "--noise-levels",
            args.noise_levels,
            "--missing-rates",
            args.missing_rates,
            "--repeats",
            str(args.noise_repeats),
        ],
        cwd=root,
    )


def main() -> None:
    args = parse_args()
    set_publication_style()
    root = Path(".").resolve()
    out_dir = Path(args.out_dir)
    work = Path(args.intermediate_dir) if args.intermediate_dir else out_dir / "_intermediate"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    data_dir = out_dir / "figure_data"
    for p in [out_dir, work, tables_dir, figures_dir, data_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if not args.skip_run:
        run_analyses(root=root, work=work, args=args)

    manifest: List[Dict[str, str]] = []

    def reg(item_id: str, item_type: str, file_path: Path, title: str, source: str, data_path: Path | None = None) -> None:
        manifest.append(
            {
                "item_id": item_id,
                "item_type": item_type,
                "priority": "main",
                "file": str(file_path.relative_to(out_dir)),
                "data_csv": str(data_path.relative_to(out_dir)) if data_path else "",
                "title_en": title,
                "source": source,
            }
        )

    # Tables (core 5)
    t1 = build_table_01(Path(args.results_csv), Path(args.raw_features_csv), Path(args.clean_features_csv), args.target_col)
    t1_path = tables_dir / "Table_01_dataset_flow_and_basic_stats.csv"
    t1.to_csv(t1_path, index=False, encoding="utf-8")
    reg("Table_01", "table", t1_path, "Dataset flow and basic sample statistics", "results/raw/clean")

    t2_path = tables_dir / "Table_02_main_model_and_baseline_performance.csv"
    t2 = copy_csv(work / "paper_main" / "cv_summary.csv", t2_path)
    t2 = t2.sort_values(["cv", "model", "feature_set"]).reset_index(drop=True)
    t2.to_csv(t2_path, index=False, encoding="utf-8")
    reg("Table_02", "table", t2_path, "Main model and baseline performance", "paper_main/cv_summary.csv")

    t3_path = tables_dir / "Table_03_hypothesis_tests_summary.csv"
    copy_csv(work / "hypothesis" / "hypothesis_summary.csv", t3_path)
    reg("Table_03", "table", t3_path, "Hypothesis tests summary", "hypothesis/hypothesis_summary.csv")

    t4_path = tables_dir / "Table_04_h2_partial_corr_coordination.csv"
    copy_csv(work / "hypothesis" / "h2_partial_corr_coordination.csv", t4_path)
    reg("Table_04", "table", t4_path, "Partial correlations for coordination features", "hypothesis/h2_partial_corr_coordination.csv")

    qc_json = work / "qc" / "qc_failure_and_bias_stats.json"
    if not qc_json.exists():
        raise FileNotFoundError(f"Missing required json: {qc_json}")
    payload = json.loads(qc_json.read_text(encoding="utf-8"))
    flat: List[Dict[str, object]] = []
    for k, v in payload.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat.append({"key": f"{k}.{kk}", "value": vv})
        else:
            flat.append({"key": k, "value": v})
    t5_path = tables_dir / "Table_05_qc_failure_and_selection_bias_stats.csv"
    pd.DataFrame(flat).to_csv(t5_path, index=False, encoding="utf-8")
    reg("Table_05", "table", t5_path, "QC failure and selection-bias statistics", "qc/qc_failure_and_bias_stats.json")

    # Figures (core 5)
    ext = args.figure_format
    f1 = figures_dir / f"Figure_01_main_model_comparison.{ext}"
    d1 = data_dir / "Figure_01_main_model_comparison_data.csv"
    fig_main_performance(work / "paper_main" / "cv_summary.csv", f1, d1, args.figure_dpi)
    reg("Figure_01", "figure", f1, "Main model comparison (RF, GroupKFold)", "paper_main/cv_summary.csv", d1)

    f2 = figures_dir / f"Figure_02_qc_threshold_tradeoff.{ext}"
    d2 = data_dir / "Figure_02_qc_threshold_tradeoff_data.csv"
    fig_qc_tradeoff(work / "qc" / "qc_threshold_sensitivity.csv", f2, d2, args.figure_dpi)
    reg("Figure_02", "figure", f2, "QC threshold tradeoff", "qc/qc_threshold_sensitivity.csv", d2)

    f3 = figures_dir / f"Figure_03_feature_importance_positive.{ext}"
    d3 = data_dir / "Figure_03_feature_importance_positive_data.csv"
    fig_importance_positive(work / "paper_main" / "feature_importance_main.csv", f3, d3, args.figure_dpi, args.topk)
    reg("Figure_03", "figure", f3, "Positive feature importance", "paper_main/feature_importance_main.csv", d3)

    f4 = figures_dir / f"Figure_04_robustness_gender_bmi.{ext}"
    d4 = data_dir / "Figure_04_robustness_gender_bmi_data.csv"
    fig_robust_gender_bmi(work / "robustness" / "subgroup_metrics.csv", f4, d4, args.figure_dpi)
    reg("Figure_04", "figure", f4, "Robustness across gender/BMI", "robustness/subgroup_metrics.csv", d4)

    f5 = figures_dir / f"Figure_05_noise_missing_heatmap.{ext}"
    d5 = data_dir / "Figure_05_noise_missing_heatmap_data.csv"
    fig_noise_missing(work / "noise" / "noise_missing_summary.csv", f5, d5, args.figure_dpi)
    reg("Figure_05", "figure", f5, "Noise/missingness sensitivity heatmap", "noise/noise_missing_summary.csv", d5)

    mf = pd.DataFrame(manifest)
    mf_path = out_dir / "asset_manifest.csv"
    mf.to_csv(mf_path, index=False, encoding="utf-8")
    print(f"Saved output dir: {out_dir}")
    print(f"Saved manifest: {mf_path}")
    print(mf.to_string(index=False))


if __name__ == "__main__":
    main()
