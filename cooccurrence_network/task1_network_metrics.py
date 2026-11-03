from __future__ import annotations

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


THRESHOLD_SURVIVE = 0.3
METRIC_META = {
    "family_diversity": ("科级多样性（P > 0.3的科数）", "family diversity"),
    "edge_count": ("共现网络边数", "edge count"),
    "density": ("共现网络密度", "density"),
    "modularity": ("共现网络模块度", "modularity"),
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
        data = raw.iloc[header_index + 1:].copy()
        layer_ids = data.iloc[:, id_index].dropna()
        layer_ids = [normalize_id(value) for value in layer_ids if normalize_id(value)]
        for seq, layer_id in enumerate(layer_ids, start=1):
            rows.append(
                {
                    "basin": str(sheet),
                    "layer_seq": seq,
                    "layer_id": layer_id,
                }
            )
    layers = pd.DataFrame(rows)
    if layers.empty:
        raise ValueError("补充S data-3未读取到有效层位。")
    layers["j3_marker"] = layers["layer_id"].str.contains("J3", case=False, na=False)
    layers["j3_base_seq"] = layers.groupby("basin")["layer_seq"].transform(
        lambda values: values.where(layers.loc[values.index, "j3_marker"]).max()
    )
    layers["period"] = np.where(layers["layer_seq"] <= layers["j3_base_seq"], "J3", "J2")
    return layers


def load_layer_probabilities(root_dir: Path, layers: pd.DataFrame) -> pd.DataFrame:
    path = root_dir / "maxent_model" / "results" / "layer_probabilities_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"未找到MaxEnt分层预测表: {path}")
    probability = pd.read_csv(path)
    family_col = "family_unit" if "family_unit" in probability.columns else "taxon"
    probability = probability[
        ["s3_sheet", "layer_seq", "layer_id", "suitability_probability", family_col]
    ].rename(columns={family_col: "family"})
    probability["basin"] = probability["s3_sheet"].astype(str)
    probability["layer_key"] = probability["layer_id"].map(normalize_id)
    layers["layer_key"] = layers["layer_id"].map(normalize_id)
    merged = probability.merge(
        layers[["basin", "layer_key", "layer_seq", "period"]],
        on=["basin", "layer_key"],
        how="inner",
        suffixes=("", "_supplemental"),
    )
    if merged.empty:
        raise ValueError("补充S data-3与MaxEnt分层预测没有匹配层位。")
    return merged


def build_presence_matrix(root_dir: Path) -> pd.DataFrame:
    sdata1 = root_dir / "data" / "dataset_001.xlsx"
    if sdata1 is None:
        raise FileNotFoundError("未找到生态六 S data-1 文件。")

    site_to_families: dict[str, set[str]] = {}
    all_families: set[str] = set()
    for sheet in pd.ExcelFile(sdata1).sheet_names:
        data = pd.read_excel(sdata1, sheet_name=sheet, usecols="A:F")
        data.columns = ["sample_number", "order", "family", "genera", "species", "taphonomic_grade"]
        families = set(
            data["family"].astype(str).str.strip()
            .replace({"nan": "", "unknown": ""})
            .loc[lambda values: values.ne("")]
        )
        site_to_families[str(sheet)] = families
        all_families.update(families)

    presence = pd.DataFrame(0, index=sorted(site_to_families), columns=sorted(all_families), dtype=int)
    for site, families in site_to_families.items():
        if families:
            presence.loc[site, list(families)] = 1
    return presence


def phi_coefficient(a: int, b: int, c: int, d: int) -> float:
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return float((a * d - b * c) / denominator) if denominator else 0.0


def build_overall_network(presence: pd.DataFrame) -> nx.Graph:
    from scipy.stats import fisher_exact

    graph = nx.Graph()
    families = presence.columns.tolist()
    graph.add_nodes_from(families)
    matrix = presence.to_numpy()
    for left in range(len(families)):
        for right in range(left + 1, len(families)):
            x, y = matrix[:, left], matrix[:, right]
            a = int(((x == 1) & (y == 1)).sum())
            b = int(((x == 1) & (y == 0)).sum())
            c = int(((x == 0) & (y == 1)).sum())
            d = int(((x == 0) & (y == 0)).sum())
            if a == 0:
                continue
            _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
            phi = phi_coefficient(a, b, c, d)
            if p_value < 0.05 and phi > 0:
                graph.add_edge(families[left], families[right], phi=phi, p_value=p_value)
    return graph


def calculate_graph_metrics(graph: nx.Graph) -> dict[str, float | int]:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    modularity = np.nan
    if edge_count > 0 and node_count > 2:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
        if communities:
            modularity = float(nx.algorithms.community.modularity(graph, communities))
    return {
        "family_diversity": int(node_count),
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "density": float(nx.density(graph)) if node_count > 1 else 0.0,
        "modularity": modularity,
        "community_count": len(communities) if edge_count > 0 and node_count > 2 and communities else 0,
    }


def calculate_layer_metrics(probability: pd.DataFrame, base_graph: nx.Graph) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (basin, seq, layer_id, period), group in probability.groupby(
        ["basin", "layer_seq", "layer_id", "period"], sort=False
    ):
        survivors = set(
            group.loc[group["suitability_probability"] >= THRESHOLD_SURVIVE, "family"].astype(str)
        )
        graph = base_graph.subgraph(survivors).copy()
        rows.append(
            {
                "basin": basin,
                "layer_seq": int(seq),
                "layer_id": layer_id,
                "period": period,
                **calculate_graph_metrics(graph),
            }
        )
    return pd.DataFrame(rows).sort_values(["basin", "layer_seq"], kind="stable").reset_index(drop=True)


def plot_metric_panels(metrics: pd.DataFrame, figure_dir: Path) -> None:
    basins = metrics["basin"].drop_duplicates().tolist()
    for metric, (label_cn, label_en) in METRIC_META.items():
        fig, axes = plt.subplots(3, 2, figsize=(13, 14), squeeze=False)
        for axis, basin in zip(axes.ravel(), basins):
            subset = metrics[metrics["basin"] == basin].sort_values("layer_seq")
            axis.set_facecolor("#E7E7E7")
            marker = subset.loc[subset["period"].eq("J3"), "layer_seq"].max()
            if pd.notna(marker) and marker < subset["layer_seq"].max():
                axis.axvspan(marker + 0.5, subset["layer_seq"].max() + 0.5, color="#D9D9D9", alpha=0.8, zorder=0)
                axis.axvspan(0.5, marker + 0.5, color="#73A9C2", alpha=0.75, zorder=0)
            axis.plot(
                subset["layer_seq"],
                subset[metric],
                color="#2C73B9",
                marker="o",
                markersize=3,
                linewidth=1.3,
                zorder=2,
            )
            axis.set_title(str(basin), fontsize=12, fontweight="bold")
            axis.set_xlabel("Layers sequence（按补充S data-3原始No.顺序）")
            axis.set_ylabel(label_cn)
            axis.grid(alpha=0.25, linestyle="--", linewidth=0.6, zorder=1)
        for axis in axes.ravel()[len(basins):]:
            axis.axis("off")
        fig.suptitle(f"{label_cn}（{label_en}）", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(figure_dir / f"task1_{metric}_by_basin.pdf", format="pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"task1_{metric}_by_basin.png", dpi=300, bbox_inches="tight")
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
    probability = load_layer_probabilities(root_dir, layers)
    presence = build_presence_matrix(root_dir)
    base_graph = build_overall_network(presence)
    metrics = calculate_layer_metrics(probability, base_graph)

    metrics.to_csv(output_dir / "task1_layer_network_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.melt(
        id_vars=["basin", "layer_seq", "layer_id", "period"],
        value_vars=list(METRIC_META),
        var_name="metric",
        value_name="value",
    ).to_csv(output_dir / "task1_plot_data_long.csv", index=False, encoding="utf-8-sig")
    plot_metric_panels(metrics, figure_dir)

    methods = """# 任务1：不同层位共现网络指标

本次重跑输入文件：`S data-3生态六地层化学数据1007.xlsx`。

## 数据与层位顺序
- 层位顺序来自本次输入文件每个sheet中 `No.` 列的原始行顺序。
- 生态六已有MaxEnt分层适生概率用于判断每个层位的科级单元是否存活。
- 存活阈值为 `suitability_probability >= 0.3`。

## 指标
- 科级多样性：该层位中适生概率达到阈值的科级单元数。
- 边数：整体科级共现网络诱导子网络的边数。
- 密度：诱导子网络的NetworkX图密度。
- 模块度：使用greedy modularity communities计算。

## 图件
- 每个指标单独生成一个PDF和PNG，图内包含5个盆地子图，保存于`figures_1007`。
- 蓝色背景为J3层位范围，灰色背景为J2层位范围；分界依据补充数据中标注J3的最低层位。
"""
    (output_dir / "methods.md").write_text(methods, encoding="utf-8")
    print("任务1完成。")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
