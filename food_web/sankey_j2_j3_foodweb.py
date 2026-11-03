from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import pandas as pd


THRESHOLD = 0.3
PERIOD_J3 = "J3"
PERIOD_J2 = "J2"
PERIOD_ORDER = [PERIOD_J2, PERIOD_J3]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def find_source_dir(root_dir: Path) -> Path:
    data_dir = root_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"未找到上传包数据目录: {data_dir}")
    return data_dir


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value).strip().lower())


def format_functional_group(value: object) -> str:
    if pd.isna(value) or str(value).strip() in {"", "nan", "None"}:
        return "Unknown functional group"
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(number) and float(number).is_integer():
        return f"FG {int(number)}"
    return f"FG {str(value).strip()}"


def load_foodweb_mapping(source_dir: Path) -> dict[str, str]:
    foodweb = source_dir / "dataset_004.xlsx"
    if foodweb is None:
        raise FileNotFoundError("未找到 S data-4 食物网文件。")

    raw = pd.read_excel(foodweb, sheet_name="Functional group")
    columns = {normalize_name(column): column for column in raw.columns}
    family_col = columns.get("family", raw.columns[1])
    group_col = columns.get("functionalgroup", raw.columns[2])
    mapping = {}
    for _, row in raw[[family_col, group_col]].dropna(how="all").iterrows():
        family = str(row[family_col]).strip()
        if family and family.lower() != "nan":
            mapping[family] = format_functional_group(row[group_col])
    return mapping


def assign_periods(probability_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundaries: list[dict[str, object]] = []
    period_df = probability_df.copy()

    for basin, group in period_df.groupby("s3_sheet", sort=False):
        marker = group[group["layer_id"].astype(str).str.contains("J3", case=False, na=False)]
        if marker.empty:
            raise ValueError(f"{basin} 未找到标注 J3 的最低层位。")
        j3_base_seq = int(marker["layer_seq"].max())
        base_row = marker.loc[marker["layer_seq"].idxmax()]
        boundaries.append(
            {
                "basin": basin,
                "j3_base_layer_seq": j3_base_seq,
                "j3_base_layer_id": str(base_row["layer_id"]),
                "rule": "layer_seq <= J3最低层位为J3；其下方为J2",
            }
        )

    boundary_df = pd.DataFrame(boundaries)
    boundary_map = boundary_df.set_index("basin")["j3_base_layer_seq"].to_dict()
    period_df["period"] = [
        PERIOD_J3 if int(seq) <= int(boundary_map[basin]) else PERIOD_J2
        for basin, seq in zip(period_df["s3_sheet"], period_df["layer_seq"])
    ]
    return period_df, boundary_df


def build_summary(
    probability_df: pd.DataFrame,
    family_to_group: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alive = probability_df[probability_df["suitability_probability"] > THRESHOLD].copy()
    alive["family"] = alive["taxon"].astype(str).str.strip()
    alive["functional_group"] = alive["family"].map(family_to_group).fillna(
        alive["functional_group"].map(format_functional_group)
    )
    alive["functional_group"] = alive["functional_group"].replace(
        {"nan": "Unknown functional group", "": "Unknown functional group"}
    )

    family_summary = (
        alive.groupby(["period", "family", "functional_group"], as_index=False)
        .agg(
            occurrence_count=("family", "size"),
            basin_count=("s3_sheet", "nunique"),
            layer_count=("layer_id", "nunique"),
            mean_probability=("suitability_probability", "mean"),
        )
        .sort_values(["period", "occurrence_count", "family"], ascending=[True, False, True])
    )
    group_summary = (
        family_summary.groupby(["period", "functional_group"], as_index=False)
        .agg(
            occurrence_count=("occurrence_count", "sum"),
            unique_family_count=("family", "nunique"),
            basin_count=("basin_count", "max"),
        )
        .sort_values(["period", "occurrence_count"], ascending=[True, False])
    )
    return alive, family_summary, group_summary


def build_links(family_summary: pd.DataFrame) -> pd.DataFrame:
    period_links = family_summary.rename(
        columns={"period": "source", "family": "target", "occurrence_count": "value"}
    )[["source", "target", "value"]].copy()
    period_links["link_type"] = "period_to_family"

    group_links = (
        family_summary.groupby(["family", "functional_group"], as_index=False)["occurrence_count"]
        .sum()
        .rename(columns={"family": "source", "functional_group": "target", "occurrence_count": "value"})
    )
    group_links["link_type"] = "family_to_functional_group"
    return pd.concat([period_links, group_links], ignore_index=True)


def node_layout(items: list[str], values: dict[str, float], bottom: float = 0.06, top: float = 0.94) -> dict[str, tuple[float, float]]:
    total = sum(values.get(item, 0.0) for item in items) or 1.0
    usable = top - bottom
    gap = min(0.012, usable / max(100.0, len(items) * 8.0))
    scale = max(0.0, (usable - gap * max(0, len(items) - 1)) / total)
    layout: dict[str, tuple[float, float]] = {}
    cursor = bottom
    for item in items:
        height = values.get(item, 0.0) * scale
        layout[item] = (cursor, cursor + height)
        cursor += height + gap
    return layout


def ribbon_path(x0: float, x1: float, y0: float, y1: float, h: float) -> PathPatch:
    bend = (x1 - x0) * 0.45
    vertices = [
        (x0, y0),
        (x0 + bend, y0),
        (x1 - bend, y1),
        (x1, y1),
        (x1, y1 + h),
        (x1 - bend, y1 + h),
        (x0 + bend, y0 + h),
        (x0, y0 + h),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(vertices, codes), linewidth=0)


def draw_sankey(
    links: pd.DataFrame,
    family_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    family_values = family_summary.groupby("family")["occurrence_count"].sum().to_dict()
    group_values = group_summary.groupby("functional_group")["occurrence_count"].sum().to_dict()
    families = sorted(family_values, key=lambda item: (-family_values[item], item))
    groups = sorted(group_values, key=lambda item: (-group_values[item], item))
    period_values = {
        period: float(family_summary.loc[family_summary["period"] == period, "occurrence_count"].sum())
        for period in PERIOD_ORDER
    }
    layout = {
        "period": {
            PERIOD_J3: (0.66, 0.66 + 0.18),
            PERIOD_J2: (0.16, 0.16 + 0.18),
        },
        "family": node_layout(families, family_values),
        "group": node_layout(groups, group_values),
    }
    group_colors = {
        group: plt.get_cmap("tab20")(index % 20)
        for index, group in enumerate(groups)
    }

    fig, ax = plt.subplots(figsize=(22, 14))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x_positions = {"period": 0.02, "family": 0.46, "group": 0.90}
    flow_cursors = {
        ("period", period): layout["period"][period][0]
        for period in PERIOD_ORDER
    }
    flow_cursors.update({("family", family): layout["family"][family][0] for family in families})
    flow_cursors.update({("group", group): layout["group"][group][0] for group in groups})
    link_rows = links.sort_values(["link_type", "source", "target"], kind="stable")

    for _, link in link_rows.iterrows():
        if link["link_type"] == "period_to_family":
            source_type, target_type = "period", "family"
        else:
            source_type, target_type = "family", "group"
        source, target = str(link["source"]), str(link["target"])
        value = float(link["value"])
        source_y = flow_cursors[(source_type, source)]
        target_y = flow_cursors[(target_type, target)]
        source_total = period_values[source] if source_type == "period" else family_values[source]
        target_total = family_values[target] if target_type == "family" else group_values[target]
        source_height = value / source_total * (layout[source_type][source][1] - layout[source_type][source][0])
        target_height = value / target_total * (layout[target_type][target][1] - layout[target_type][target][0])
        height = min(source_height, target_height)
        patch = ribbon_path(x_positions[source_type] + 0.025, x_positions[target_type] - 0.025, source_y, target_y, height)
        if source_type == "period":
            family_group = family_summary.loc[
                family_summary["family"] == target, "functional_group"
            ].iloc[0]
        else:
            family_group = target
        color = group_colors.get(family_group, "gray")
        patch.set_facecolor(color)
        patch.set_alpha(0.34 if source_type == "period" else 0.55)
        ax.add_patch(patch)
        flow_cursors[(source_type, source)] += source_height
        flow_cursors[(target_type, target)] += target_height

    for period in PERIOD_ORDER:
        y0, y1 = layout["period"][period]
        color = "#4C78A8" if period == PERIOD_J2 else "#D95F02"
        ax.add_patch(Rectangle((x_positions["period"], y0), 0.025, y1 - y0, facecolor=color, edgecolor="white"))
        ax.text(x_positions["period"] - 0.008, (y0 + y1) / 2, period, ha="right", va="center", fontsize=14, fontweight="bold")

    for family in families:
        y0, y1 = layout["family"][family]
        group = str(family_summary.loc[family_summary["family"] == family, "functional_group"].iloc[0])
        ax.add_patch(Rectangle((x_positions["family"], y0), 0.018, y1 - y0, facecolor=group_colors[group], edgecolor="white", linewidth=0.3))
        if family_values[family] >= np.quantile(list(family_values.values()), 0.9):
            ax.text(x_positions["family"] + 0.023, (y0 + y1) / 2, family, ha="left", va="center", fontsize=6)

    for group in groups:
        y0, y1 = layout["group"][group]
        ax.add_patch(Rectangle((x_positions["group"], y0), 0.025, y1 - y0, facecolor=group_colors[group], edgecolor="white"))
        ax.text(x_positions["group"] + 0.032, (y0 + y1) / 2, group, ha="left", va="center", fontsize=8)

    ax.text(x_positions["period"], 0.985, "时代组", ha="left", va="bottom", fontsize=12, fontweight="bold")
    ax.text(x_positions["family"], 0.985, "科级分类单元（仅标注前10%）", ha="left", va="bottom", fontsize=12, fontweight="bold")
    ax.text(x_positions["group"], 0.985, "食物网功能群", ha="left", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("J2 与 J3 科级多样性—食物网功能群桑基图", fontsize=18, pad=24)
    fig.text(0.5, 0.02, "流量 = 所有盆地、所有层位中适生概率 > 0.3 的科级单元出现次数；J2位于下方，J3位于上方", ha="center", fontsize=10)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def variable_ribbon_path(
    x0: float,
    x1: float,
    source_y: float,
    source_height: float,
    target_y: float,
    target_height: float,
) -> PathPatch:
    bend = (x1 - x0) * 0.45
    vertices = [
        (x0, source_y),
        (x0 + bend, source_y),
        (x1 - bend, target_y),
        (x1, target_y),
        (x1, target_y + target_height),
        (x1 - bend, target_y + target_height),
        (x0 + bend, source_y + source_height),
        (x0, source_y + source_height),
        (x0, source_y),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return PathPatch(MplPath(vertices, codes), linewidth=0)


def build_diversity_table(alive: pd.DataFrame, category_col: str) -> pd.DataFrame:
    summary = (
        alive.groupby(["period", category_col], as_index=False)["family"]
        .nunique()
        .rename(columns={"family": "family_diversity"})
    )
    table = summary.pivot(index=category_col, columns="period", values="family_diversity").fillna(0)
    for period in PERIOD_ORDER:
        if period not in table.columns:
            table[period] = 0
    table = table.reset_index()
    table["J2"] = table[PERIOD_J2].astype(int)
    table["J3"] = table[PERIOD_J3].astype(int)
    table["change_J3_minus_J2"] = table["J3"] - table["J2"]
    table["change_percent"] = np.where(
        table["J2"] > 0,
        table["change_J3_minus_J2"] / table["J2"] * 100,
        np.nan,
    )
    return table[[category_col, "J2", "J3", "change_J3_minus_J2", "change_percent"]].sort_values(
        ["J2", "J3", category_col], ascending=[False, False, True]
    )


def draw_diversity_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    category_col: str,
    title: str,
) -> None:
    categories = table[category_col].astype(str).tolist()
    j2_values = dict(zip(categories, table["J2"].astype(float)))
    j3_values = dict(zip(categories, table["J3"].astype(float)))
    visible = [category for category in categories if j2_values[category] > 0 or j3_values[category] > 0]
    total_j2 = sum(j2_values.values()) or 1.0
    total_j3 = sum(j3_values.values()) or 1.0
    bottom, top, gap = 0.08, 0.92, 0.006
    scale_j2 = (top - bottom - gap * max(0, len(visible) - 1)) / total_j2
    scale_j3 = (top - bottom - gap * max(0, len(visible) - 1)) / total_j3
    source_layout: dict[str, tuple[float, float]] = {}
    target_layout: dict[str, tuple[float, float]] = {}
    cursor_j2 = bottom
    cursor_j3 = bottom
    for category in visible:
        h2 = j2_values[category] * scale_j2
        h3 = j3_values[category] * scale_j3
        source_layout[category] = (cursor_j2, h2)
        target_layout[category] = (cursor_j3, h3)
        cursor_j2 += h2 + gap
        cursor_j3 += h3 + gap

    colors = {
        category: plt.get_cmap("tab20")(index % 20)
        for index, category in enumerate(visible)
    }
    x_j2, x_j3 = 0.19, 0.77
    bar_width = 0.045
    for category in visible:
        y2, h2 = source_layout[category]
        y3, h3 = target_layout[category]
        patch = variable_ribbon_path(x_j2 + bar_width, x_j3, y2, h2, y3, h3)
        patch.set_facecolor(colors[category])
        patch.set_alpha(0.55)
        ax.add_patch(patch)

    for x, label, layout in [(x_j2, "J2", source_layout), (x_j3, "J3", target_layout)]:
        ymin = bottom
        ymax = top
        ax.add_patch(
            Rectangle(
                (x, ymin),
                bar_width,
                ymax - ymin,
                facecolor="white",
                edgecolor="#D9485F",
                linewidth=2.5,
                zorder=5,
            )
        )
        ax.text(x + bar_width / 2, ymax + 0.025, label, ha="center", va="bottom", fontsize=14, color="#D9485F", fontweight="bold")

    label_limit = min(8, len(visible))
    for category in visible[:label_limit]:
        y2, h2 = source_layout[category]
        y3, h3 = target_layout[category]
        if h2 > 0.012:
            ax.text(x_j2 - 0.012, y2 + h2 / 2, category, ha="right", va="center", fontsize=7)
        if h3 > 0.012:
            ax.text(x_j3 + bar_width + 0.012, y3 + h3 / 2, category, ha="left", va="center", fontsize=7)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=15, color="#D9485F", pad=18)


def draw_diversity_comparison(
    order_table: pd.DataFrame,
    group_table: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    draw_diversity_panel(axes[0], order_table, "order", "比较目下科的多样性")
    draw_diversity_panel(
        axes[1],
        group_table,
        "functional_group",
        "比较各功能群的科多样性变化",
    )
    fig.text(0.035, 0.5, "科级多样性（科数）", rotation=90, ha="center", va="center", fontsize=13, color="#D9485F")
    fig.text(
        0.5,
        0.025,
        "所有盆地合并；每条带宽表示该目/功能群在对应时代中包含的唯一科级分类单元数量",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.05, 0.04, 0.98, 0.96])
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_summary_sankey(group_summary: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    links = group_summary.rename(columns={"period": "source", "functional_group": "target", "occurrence_count": "value"})
    links = links[["source", "target", "value"]].copy()
    links["link_type"] = "period_to_group"
    family_like = group_summary.rename(columns={"functional_group": "family", "occurrence_count": "value"}).copy()
    family_like["functional_group"] = family_like["family"]
    family_like["occurrence_count"] = family_like["value"]
    family_like["family"] = family_like["functional_group"]
    draw_sankey(links.rename(columns={"target": "target"}), family_like, group_summary, out_pdf, out_png)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[2]
    source_dir = find_source_dir(root_dir)
    output_dir = root_dir / "sankey_j2_j3"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    probability_path = root_dir / "maxent_model" / "results" / "layer_probabilities_long.csv"
    if not probability_path.exists():
        raise FileNotFoundError("未找到 outputs_maxent/layer_probabilities_long.csv，请先运行MaxEnt。")

    probability_df = pd.read_csv(probability_path)
    probability_df, boundary_df = assign_periods(probability_df)
    family_to_group = load_foodweb_mapping(source_dir)
    alive, family_summary, group_summary = build_summary(probability_df, family_to_group)
    links = build_links(family_summary)
    order_table = build_diversity_table(alive, "order")
    group_table = build_diversity_table(alive, "functional_group")

    boundary_df.to_csv(output_dir / "j2_j3_boundary_layers.csv", index=False, encoding="utf-8-sig")
    family_summary.to_csv(output_dir / "j2_j3_family_diversity_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(output_dir / "j2_j3_functional_group_summary.csv", index=False, encoding="utf-8-sig")
    order_table.to_csv(output_dir / "j2_j3_order_family_diversity.csv", index=False, encoding="utf-8-sig")
    group_table.to_csv(output_dir / "j2_j3_functional_group_family_diversity.csv", index=False, encoding="utf-8-sig")
    links.to_csv(output_dir / "j2_j3_sankey_links.csv", index=False, encoding="utf-8-sig")

    draw_sankey(
        links,
        family_summary,
        group_summary,
        figure_dir / "sankey_j2_j3_family_functional_group.pdf",
        figure_dir / "sankey_j2_j3_family_functional_group.png",
    )
    draw_diversity_comparison(
        order_table,
        group_table,
        figure_dir / "sankey_j2_j3_diversity_comparison.pdf",
        figure_dir / "sankey_j2_j3_diversity_comparison.png",
    )
    methods = """# J2-J3桑基图方法

## 分组规则
- 每个盆地中，层位编号包含 `(J3)` 或 `J3` 的记录视为晚侏罗世最低层位。
- 按Excel原始行顺序，标记层及其上方（行序号更小、地层更新）归入J3。
- 标记层下方（行序号更大、地层更老）归入J2。
- 桑基图中J2放在下方，J3放在上方。

## 流量定义
- 科级单元：`suitability_probability > 0.3` 的科级单元。
- 时代组 → 科级分类单元：该科在所有盆地、所有对应时代层位中的出现次数。
- 科级分类单元 → 功能群：按S data-4的Functional group表归属，流量保持守恒。
- 未能在S data-4中匹配的科归入Unknown functional group。

## 输出
- `j2_j3_boundary_layers.csv`：各盆地J3最低层位及分组边界。
- `j2_j3_family_diversity_summary.csv`：时代-科-功能群汇总。
- `j2_j3_functional_group_summary.csv`：时代-功能群汇总。
- `j2_j3_order_family_diversity.csv`：各目下唯一科级单元数的J2/J3比较。
- `j2_j3_functional_group_family_diversity.csv`：各功能群下唯一科级单元数的J2/J3比较。
- `j2_j3_sankey_links.csv`：桑基图边表。
- `figures/sankey_j2_j3_family_functional_group.pdf/png`：桑基图。
- `figures/sankey_j2_j3_diversity_comparison.pdf/png`：按示意图样式的两个J2/J3对比面板。
"""
    (output_dir / "methods.md").write_text(methods, encoding="utf-8")
    print("J2-J3食物网桑基图生成完成。")
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
