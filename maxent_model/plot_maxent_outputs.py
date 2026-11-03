from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensureDir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def loadCsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    return pd.read_csv(path)


def safeFileName(text: str) -> str:
    chars = []
    for ch in str(text):
        if ch.isalnum() or ch in ["-", "_"]:
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)[:100]


def saveFigurePdf(out_path: Path) -> Path:
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    return out_path


def loadTaxonomyReference(base_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str], List[str], List[str]]:
    data_path = base_dir.parent / "data" / "dataset_004.xlsx"
    if not data_path.exists():
        return pd.DataFrame(columns=["order", "family"]), {}, [], []
    try:
        df = pd.read_excel(data_path, sheet_name="Functional group")
    except Exception:
        df = pd.read_excel(data_path, sheet_name=0)
    if df.empty:
        return pd.DataFrame(columns=["order", "family"]), {}, [], []

    cols = list(df.columns)
    order_col = next((c for c in cols if str(c).strip().lower() in {"?", "order"}), cols[0])
    family_col = next((c for c in cols if str(c).strip().lower() in {"?", "family"}), cols[1] if len(cols) > 1 else cols[0])
    ref = df[[order_col, family_col]].copy()
    ref.columns = ["order", "family"]
    ref["order"] = ref["order"].astype(str).str.strip()
    ref["family"] = ref["family"].astype(str).str.strip()
    ref = ref[(ref["order"].ne("")) & (ref["order"].ne("nan")) & (ref["family"].ne("")) & (ref["family"].ne("nan"))].drop_duplicates()
    family_to_order = dict(zip(ref["family"], ref["order"]))
    return ref, family_to_order, ref["family"].tolist(), ref["order"].drop_duplicates().tolist()
def getFamilyColumn(df: pd.DataFrame) -> str:
    return "family_unit" if "family_unit" in df.columns else "taxon"


def plotAucBar(metrics_df: pd.DataFrame, fig_dir: Path, family_order: List[str]) -> Path:
    family_col = getFamilyColumn(metrics_df)
    valid = metrics_df[metrics_df["auc"].notna()].copy()
    order_rank = {v: i for i, v in enumerate(family_order)}
    valid["__rank"] = valid[family_col].map(order_rank).fillna(10**9)
    valid = valid.sort_values(["__rank", family_col], kind="stable")

    fig_h = min(45, max(8, 0.28 * len(valid)))
    plt.figure(figsize=(11, fig_h))
    plt.barh(valid[family_col], valid["auc"], color="#4C78A8")
    plt.xlim(0, 1.02)
    plt.xlabel("AUC")
    plt.ylabel("Family Unit")
    plt.title("AUC for All Family-level Units")
    plt.tight_layout()
    return saveFigurePdf(fig_dir / "fig_01_auc_all_family_units_bar.pdf")


def plotAucDistribution(metrics_df: pd.DataFrame, fig_dir: Path) -> Path:
    plt.figure(figsize=(8, 5))
    plt.hist(metrics_df["auc"].dropna(), bins=12, color="#59A14F", edgecolor="black")
    plt.xlabel("AUC")
    plt.ylabel("Count")
    plt.title("AUC Distribution (All Family-level Units)")
    plt.tight_layout()
    return saveFigurePdf(fig_dir / "fig_02_auc_distribution.pdf")


def plotRocAllFamilies(metrics_df: pd.DataFrame, roc_df: pd.DataFrame, fig_dir: Path) -> Path:
    family_col = getFamilyColumn(metrics_df)
    taxa_all = metrics_df[metrics_df["auc"].notna()]["taxon"].tolist()
    plt.figure(figsize=(9, 9))
    grid = np.linspace(0, 1, 200)
    curves_interp: List[np.ndarray] = []

    for taxon in taxa_all:
        sub = roc_df[roc_df["taxon"] == taxon].sort_values("fpr").drop_duplicates("fpr")
        if sub.empty:
            continue
        fpr = sub["fpr"].to_numpy(dtype=float)
        tpr = sub["tpr"].to_numpy(dtype=float)
        plt.step(fpr, tpr, where="post", linewidth=0.8, alpha=0.25, color="#2E86AB")
        curves_interp.append(np.interp(grid, fpr, tpr))

    mean_line = None
    if curves_interp:
        mean_tpr = np.mean(curves_interp, axis=0)
        mean_line = plt.plot(grid, mean_tpr, color="#D1495B", linewidth=2.2)[0]

    random_line = plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)[0]
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves for All Family-level Units (n={len(taxa_all)})")

    handles = [
        Line2D([0], [0], color="#2E86AB", lw=1.5, alpha=0.5, label=f"Family ROC curves (n={len(taxa_all)})"),
    ]
    labels = [f"Family ROC curves (n={len(taxa_all)})"]
    if mean_line is not None:
        handles.append(mean_line)
        labels.append("Mean ROC")
    handles.append(random_line)
    labels.append("Random guess")
    plt.legend(handles, labels, fontsize=9, loc="lower right", frameon=True)
    plt.tight_layout()
    return saveFigurePdf(fig_dir / "fig_03_roc_all_family_units.pdf")


def plotFeatureContribution(contrib_df: pd.DataFrame, fig_dir: Path) -> Path:
    mean_contrib = (
        contrib_df.groupby("feature", as_index=False)["contribution_percent"].mean().sort_values("contribution_percent")
    )
    plt.figure(figsize=(9, 6))
    plt.barh(mean_contrib["feature"], mean_contrib["contribution_percent"], color="#F28E2B")
    plt.xlabel("Mean Contribution (%)")
    plt.ylabel("Feature")
    plt.title("Environmental Feature Contributions (All Features)")
    plt.tight_layout()
    return saveFigurePdf(fig_dir / "fig_04_feature_contribution_all_features.pdf")


def plotLayerHeatmapBySheet(prob_wide_df: pd.DataFrame, fig_dir: Path, family_order: List[str]) -> List[Path]:
    sheet_col = "s3_sheet"
    base_cols = {sheet_col, "layer_id", "layer_seq"}
    family_cols = [c for c in prob_wide_df.columns if c not in base_cols]
    ordered_family = [f for f in family_order if f in family_cols]
    ordered_family.extend([f for f in family_cols if f not in set(ordered_family)])
    fig_paths: List[Path] = []

    for sheet_name in prob_wide_df[sheet_col].astype(str).unique():
        sub = prob_wide_df[prob_wide_df[sheet_col].astype(str) == sheet_name].copy()
        if sub.empty:
            continue
        if "layer_seq" in sub.columns:
            sub = sub.sort_values("layer_seq", kind="stable")
        mat = sub[ordered_family].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

        fig_w = min(42, max(16, 0.23 * len(ordered_family)))
        plt.figure(figsize=(fig_w, 7))
        im = plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, label="Suitability Probability")
        plt.xticks(
            range(len(ordered_family)),
            [x[:18] + ("..." if len(x) > 18 else "") for x in ordered_family],
            rotation=70,
            ha="right",
        )
        y_labels = sub["layer_id"].astype(str).tolist()
        step = max(1, len(y_labels) // 20)
        tick_idx = list(range(0, len(y_labels), step))
        plt.yticks(tick_idx, [y_labels[i] for i in tick_idx])
        plt.xlabel("Family-level Units (ordered by taxonomy list)")
        plt.ylabel(f"Layers ({sheet_name})")
        plt.title(f"Layer-wise Suitability Heatmap (All Families) - {sheet_name}")
        plt.tight_layout()
        fig_paths.append(saveFigurePdf(fig_dir / f"fig_05_heatmap_{safeFileName(sheet_name)}.pdf"))
    return fig_paths


def buildOrderDiversityLong(
    prob_long_df: pd.DataFrame,
    family_to_order: Dict[str, str],
    threshold: float = 0.3,
) -> pd.DataFrame:
    family_col = getFamilyColumn(prob_long_df)
    df = prob_long_df.copy()
    if "layer_seq" not in df.columns:
        df["layer_seq"] = df.groupby("s3_sheet").cumcount() + 1

    if "order" in df.columns:
        df["order_plot"] = df["order"].astype(str).str.strip()
        df.loc[df["order_plot"].isin(["", "nan"]), "order_plot"] = df.loc[
            df["order_plot"].isin(["", "nan"]), family_col
        ].map(family_to_order)
    else:
        df["order_plot"] = df[family_col].map(family_to_order)
    df["order_plot"] = df["order_plot"].fillna("Unknown_order")

    above = df[df["suitability_probability"] > threshold].copy()
    count_df = (
        above.groupby(["s3_sheet", "layer_seq", "layer_id", "order_plot"], as_index=False)
        .agg(family_diversity_gt_0_3=(family_col, "nunique"))
    )
    return count_df


def plotOrderDiversityHeatmapBySheet(
    prob_long_df: pd.DataFrame,
    fig_dir: Path,
    family_to_order: Dict[str, str],
    order_order: List[str],
    threshold: float = 0.3,
) -> Tuple[List[Path], pd.DataFrame]:
    count_df = buildOrderDiversityLong(prob_long_df, family_to_order, threshold=threshold)
    layer_df = (
        prob_long_df[["s3_sheet", "layer_seq", "layer_id"]]
        .drop_duplicates()
        .sort_values(["s3_sheet", "layer_seq"], kind="stable")
    )
    all_orders = count_df["order_plot"].drop_duplicates().tolist()
    ordered_orders = [o for o in order_order if o in all_orders]
    ordered_orders.extend([o for o in all_orders if o not in set(ordered_orders)])
    fig_paths: List[Path] = []

    for sheet_name in layer_df["s3_sheet"].astype(str).drop_duplicates().tolist():
        layer_sub = layer_df[layer_df["s3_sheet"].astype(str) == sheet_name].sort_values("layer_seq", kind="stable")
        pivot = pd.DataFrame(index=layer_sub["layer_seq"].tolist(), columns=ordered_orders, data=0.0)
        if not count_df.empty:
            count_sub = count_df[count_df["s3_sheet"].astype(str) == sheet_name]
            for _, row in count_sub.iterrows():
                seq = int(row["layer_seq"])
                order = str(row["order_plot"])
                if order in pivot.columns and seq in pivot.index:
                    pivot.loc[seq, order] = float(row["family_diversity_gt_0_3"])

        mat = pivot.to_numpy(dtype=float)
        fig_w = min(30, max(12, 0.28 * max(1, len(ordered_orders))))
        plt.figure(figsize=(fig_w, 7))
        im = plt.imshow(mat, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="Family diversity (probability > 0.3)")
        plt.xticks(
            range(len(ordered_orders)),
            [x[:18] + ("..." if len(x) > 18 else "") for x in ordered_orders],
            rotation=70,
            ha="right",
        )
        y_labels = layer_sub["layer_id"].astype(str).tolist()
        step = max(1, len(y_labels) // 20)
        tick_idx = list(range(0, len(y_labels), step))
        plt.yticks(tick_idx, [y_labels[i] for i in tick_idx])
        plt.xlabel("Order-level units")
        plt.ylabel(f"Layers ({sheet_name})")
        plt.title(f"Order-level Diversity Heatmap (family prob > 0.3) - {sheet_name}")
        plt.tight_layout()
        fig_paths.append(saveFigurePdf(fig_dir / f"fig_06_order_diversity_{safeFileName(sheet_name)}.pdf"))

    return fig_paths, count_df


def writeManifest(output_dir: Path, figure_files: List[Path]) -> Path:
    table_files = [
        "sdata1_sheet_mapping.csv",
        "indicator_definition.csv",
        "sdata2_indicator_values.csv",
        "model_metrics_auc.csv",
        "roc_curve_points.csv",
        "variable_contributions.csv",
        "layer_probabilities_long.csv",
        "layer_probabilities_wide.csv",
        "order_family_diversity_gt0_3_long.csv",
        "method_description.md",
    ]
    lines = [
        "# 结果文件清单",
        "",
        "## 表格与方法文件",
    ]
    for name in table_files:
        lines.append(f"- `{(output_dir / name).as_posix()}`")
    lines.append("")
    lines.append("## 绘图文件")
    for fig in figure_files:
        lines.append(f"- `{fig.as_posix()}`")
    out_path = output_dir / "result_tables_and_figures.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "results"
    fig_dir = output_dir / "figures"
    ensureDir(fig_dir)

    metrics_df = loadCsv(output_dir / "model_metrics_auc.csv")
    roc_df = loadCsv(output_dir / "roc_curve_points.csv")
    contrib_df = loadCsv(output_dir / "variable_contributions.csv")
    prob_wide_df = loadCsv(output_dir / "layer_probabilities_wide.csv")
    prob_long_df = loadCsv(output_dir / "layer_probabilities_long.csv")

    _, family_to_order, family_order, order_order = loadTaxonomyReference(base_dir)

    figure_files = [
        plotAucBar(metrics_df, fig_dir, family_order),
        plotAucDistribution(metrics_df, fig_dir),
        plotRocAllFamilies(metrics_df, roc_df, fig_dir),
        plotFeatureContribution(contrib_df, fig_dir),
    ]
    figure_files.extend(plotLayerHeatmapBySheet(prob_wide_df, fig_dir, family_order))
    order_figs, order_diversity_df = plotOrderDiversityHeatmapBySheet(
        prob_long_df, fig_dir, family_to_order, order_order, threshold=0.3
    )
    figure_files.extend(order_figs)
    order_diversity_df.to_csv(output_dir / "order_family_diversity_gt0_3_long.csv", index=False, encoding="utf-8-sig")

    manifest = writeManifest(output_dir, figure_files)

    print("绘图完成。")
    print(f"图目录: {fig_dir}")
    print(f"清单文件: {manifest}")


if __name__ == "__main__":
    main()


