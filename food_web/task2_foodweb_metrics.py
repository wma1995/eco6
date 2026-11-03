from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_META = {
    "secondary_extinction_rate": ("次级灭绝率（%）", "secondary extinction rate"),
    "predator_diversity": ("捕食者多样性（科数）", "predator diversity"),
    "trophic_link_count": ("营养连接数", "trophic link count"),
    "trophic_link_turnover_rate": ("营养连接数周转率（%）", "trophic link turnover rate"),
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def find_dir_with_codepoint(parent: Path, codepoint: int) -> Path:
    matches = [
        path for path in parent.iterdir()
        if path.is_dir() and any(ord(char) == codepoint for char in path.name)
    ]
    if not matches:
        raise FileNotFoundError(f"未找到目标目录: U+{codepoint:04X}")
    return matches[0]


def normalize_id(value: object) -> str:
    return re.sub(r"[\s\u200b\uFEFF]+", "", str(value)).strip()


def load_supplemental_layers(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sheet in pd.ExcelFile(path).sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        header_rows = [
            index for index, row in raw.iterrows()
            if row.astype(str).str.strip().eq("No.").any()
        ]
        if not header_rows:
            raise ValueError(f"{sheet} 未找到 No. 列标题。")
        header_index = header_rows[0]
        header = raw.iloc[header_index].astype(str).str.strip()
        id_index = int(header[header.eq("No.")].index[0])
        layer_ids = raw.iloc[header_index + 1:, id_index].dropna()
        layer_ids = [normalize_id(value) for value in layer_ids if normalize_id(value)]
        for seq, layer_id in enumerate(layer_ids, start=1):
            rows.append({"basin": str(sheet), "layer_seq": seq, "layer_id": layer_id})
    layers = pd.DataFrame(rows)
    layers["j3_marker"] = layers["layer_id"].str.contains("J3", case=False, na=False)
    layers["j3_base_seq"] = layers.groupby("basin")["layer_seq"].transform(
        lambda values: values.where(layers.loc[values.index, "j3_marker"]).max()
    )
    layers["period"] = np.where(layers["layer_seq"] <= layers["j3_base_seq"], "J3", "J2")
    layers["layer_key"] = layers["layer_id"].map(normalize_id)
    return layers


def load_probability_data(root_dir: Path, layers: pd.DataFrame) -> pd.DataFrame:
    path = root_dir / "maxent_model" / "results" / "layer_probabilities_long.csv"
    probability = pd.read_csv(path)
    family_col = "family_unit" if "family_unit" in probability.columns else "taxon"
    probability = probability[
        ["s3_sheet", "layer_id", "suitability_probability", family_col]
    ].rename(columns={family_col: "family"})
    probability["basin"] = probability["s3_sheet"].astype(str)
    probability["layer_key"] = probability["layer_id"].map(normalize_id)
    return probability.merge(
        layers[["basin", "layer_key", "layer_seq", "period"]],
        on=["basin", "layer_key"],
        how="inner",
    )


def load_foodweb_functions(root_dir: Path):
    foodweb = root_dir / "data" / "dataset_004.xlsx"
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from foodweb_ceg_layer_analysis import (
        build_family_graph,
        build_adjacency,
        cascade_secondary_extinction,
        edge_set_for_alive,
        load_foodweb_reference,
    )

    if not foodweb.exists():
        raise FileNotFoundError(f"缺少上传包数据文件: {foodweb}")
    fg_map, fg_edges = load_foodweb_reference(foodweb)
    return (
        build_family_graph,
        build_adjacency,
        cascade_secondary_extinction,
        edge_set_for_alive,
        fg_map,
        fg_edges,
    )


def jaccard_similarity(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def calculate_metrics(probability: pd.DataFrame, root_dir: Path) -> pd.DataFrame:
    (
        build_family_graph,
        build_adjacency,
        cascade_secondary_extinction,
        edge_set_for_alive,
        fg_map,
        fg_edges,
    ) = load_foodweb_functions(root_dir)
    modeled_families = set(probability["family"].astype(str))
    baseline_nodes, baseline_edges, _ = build_family_graph(fg_map, fg_edges, modeled_families)
    prey_map, _ = build_adjacency(baseline_nodes, baseline_edges)

    rows: list[dict[str, object]] = []
    link_sets: dict[tuple[str, int], set[tuple[str, str]]] = {}
    grouped = probability.groupby(["basin", "layer_seq", "layer_id", "period"], sort=False)
    for (basin, seq, layer_id, period), group in grouped:
        family_probability = group.groupby("family")["suitability_probability"].max()
        primary_alive = {
            family for family in baseline_nodes
            if family_probability.get(family, 0.0) >= 0.3
        }
        final_alive, secondary_extinct = cascade_secondary_extinction(primary_alive, prey_map)
        final_links = edge_set_for_alive(final_alive, baseline_edges)
        predator_count = sum(
            1 for family in final_alive
            if len(prey_map.get(family, set()) & final_alive) > 0
        )
        link_sets[(str(basin), int(seq))] = final_links
        rows.append(
            {
                "basin": str(basin),
                "layer_seq": int(seq),
                "layer_seq_plot": int(seq) - 1,
                "layer_id": str(layer_id),
                "period": period,
                "primary_alive_count": len(primary_alive),
                "secondary_extinct_count": len(secondary_extinct),
                "secondary_extinction_rate": (
                    len(secondary_extinct) / len(baseline_nodes)
                    if baseline_nodes else np.nan
                ),
                "predator_diversity": int(predator_count),
                "trophic_link_count": int(len(final_links)),
            }
        )

    result = pd.DataFrame(rows).sort_values(["basin", "layer_seq"], kind="stable")
    turnover = []
    for basin, group in result.groupby("basin", sort=False):
        previous: set[tuple[str, str]] | None = None
        for seq in group["layer_seq"]:
            current = link_sets[(basin, int(seq))]
            turnover.append(
                {
                    "basin": basin,
                    "layer_seq": int(seq),
                    "trophic_link_turnover_rate": (
                        np.nan if previous is None else 1.0 - jaccard_similarity(current, previous)
                    ),
                }
            )
            previous = current
    return result.merge(pd.DataFrame(turnover), on=["basin", "layer_seq"], how="left")


def plot_metric_panels(metrics: pd.DataFrame, figure_dir: Path) -> None:
    basins = metrics["basin"].drop_duplicates().tolist()
    for metric, (label_cn, label_en) in METRIC_META.items():
        fig, axes = plt.subplots(3, 2, figsize=(13, 14), squeeze=False)
        for axis, basin in zip(axes.ravel(), basins):
            subset = metrics[metrics["basin"] == basin].sort_values("layer_seq")
            axis.set_facecolor("#E7E7E7")
            marker = subset.loc[subset["period"].eq("J3"), "layer_seq_plot"].max()
            if pd.notna(marker) and marker < subset["layer_seq_plot"].max():
                axis.axvspan(-0.5, marker + 0.5, color="#73A9C2", alpha=0.75, zorder=0)
                axis.axvspan(marker + 0.5, subset["layer_seq_plot"].max() + 0.5, color="#D9D9D9", alpha=0.8, zorder=0)
            values = subset[metric].astype(float)
            if metric in {"secondary_extinction_rate", "trophic_link_turnover_rate"}:
                values = values * 100.0
            axis.plot(
                subset["layer_seq_plot"],
                values,
                color="#D95F5F",
                marker="o",
                markersize=3,
                linewidth=1.3,
                zorder=2,
            )
            axis.set_title(str(basin), fontsize=12, fontweight="bold")
            axis.set_xlabel("Layers sequence（从0开始，按No.原始顺序）")
            axis.set_ylabel(label_cn)
            axis.grid(alpha=0.25, linestyle="--", linewidth=0.6, zorder=1)
        for axis in axes.ravel()[len(basins):]:
            axis.axis("off")
        fig.suptitle(f"{label_cn}（{label_en}）", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(figure_dir / f"task2_{metric}_by_basin.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"task2_{metric}_by_basin.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    root_dir = task_dir.parent.parent
    supplemental = root_dir / "data" / "dataset_005.xlsx"
    if not supplemental.exists():
        raise FileNotFoundError(f"缺少上传包数据文件: {supplemental}")

    output_dir = task_dir / "results"
    figure_dir = output_dir / "figures_1007"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    layers = load_supplemental_layers(supplemental)
    probability = load_probability_data(root_dir, layers)
    metrics = calculate_metrics(probability, root_dir)
    metrics.to_csv(output_dir / "task2_layer_foodweb_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.melt(
        id_vars=["basin", "layer_seq", "layer_seq_plot", "layer_id", "period"],
        value_vars=list(METRIC_META),
        var_name="metric",
        value_name="value",
    ).to_csv(output_dir / "task2_plot_data_long.csv", index=False, encoding="utf-8-sig")
    plot_metric_panels(metrics, figure_dir)

    methods = """# 任务2：不同层位食物网指标

本次重跑输入文件：`S data-3生态六地层化学数据1007.xlsx`。

## 指标定义
- 次级灭绝率：食物网级联后次级灭绝科数 / 基准食物网科数。
- 捕食者多样性：级联稳定后仍有至少一个存活猎物的捕食者科数。
- 营养连接数：级联稳定后存活科之间的营养连接数量。
- 营养连接数周转率：当前层位与上一层位营养连接集合的 `1 - Jaccard相似度`；首层为空。

## 分组与顺序
- 使用本次1007输入文件中No.列原始顺序。
- 图中横坐标为0-based绘图序号，原始1-based `layer_seq`同时保存在CSV中。
- 蓝色背景为J3，灰色背景为J2。
- J3边界按照各盆地标注J3的最低层位确定。
- 本次PDF和PNG图件保存于`figures_1007`。

## 数据来源
- 分层科级适生概率：生态六 `outputs_maxent/layer_probabilities_long.csv`。
- 功能群及捕食关系：生态六数据源 `S data-4`。
"""
    (output_dir / "methods.md").write_text(methods, encoding="utf-8")
    print("任务2完成。")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
