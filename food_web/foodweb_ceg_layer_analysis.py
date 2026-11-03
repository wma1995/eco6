from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

THRESHOLD_SURVIVE = 0.3


@dataclass
class TPaths:
    root_dir: Path
    ref_foodweb_xlsx: Path
    maxent_long_csv: Path
    out_dir: Path
    fig_dir: Path


def detect_paths(work_dir: Path) -> TPaths:
    upload_root = work_dir.resolve().parents[1]
    foodweb = upload_root / "data" / "dataset_004.xlsx"
    maxent_long_csv = upload_root / "maxent_model" / "results" / "layer_probabilities_long.csv"
    if not foodweb.exists():
        raise FileNotFoundError(f"缺少上传包数据文件: {foodweb}")
    if not maxent_long_csv.exists():
        raise FileNotFoundError(f"请先运行最大熵模型生成: {maxent_long_csv}")
    out_dir = work_dir / "results"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return TPaths(upload_root, foodweb, maxent_long_csv, out_dir, fig_dir)

def _normalize_col_name(col: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(col).strip().lower())


def load_foodweb_reference(xlsx_path: Path) -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    xl = pd.ExcelFile(xlsx_path)
    fg_sheet = None
    matrix_sheet = None
    for s in xl.sheet_names:
        sl = s.lower()
        if "functional" in sl:
            fg_sheet = s
        if "捕食" in s or "predatory" in sl:
            matrix_sheet = s

    if fg_sheet is None:
        raise ValueError("参考食物网文件中未找到 functional group sheet。")
    if matrix_sheet is None:
        raise ValueError("参考食物网文件中未找到捕食关系 sheet。")

    df_fg_raw = pd.read_excel(xlsx_path, sheet_name=fg_sheet)
    cols = list(df_fg_raw.columns)
    norm_map = {_normalize_col_name(c): c for c in cols}

    order_col = None
    family_col = None
    fg_col = None
    for k, c in norm_map.items():
        if k in {"order", "目"}:
            order_col = c
        if k in {"family", "科"}:
            family_col = c
        if "functionalgroup" in k or "功能群" in k:
            fg_col = c
    if order_col is None:
        order_col = cols[0]
    if family_col is None:
        family_col = cols[1] if len(cols) > 1 else cols[0]
    if fg_col is None:
        fg_col = cols[2] if len(cols) > 2 else cols[-1]

    df_fg = df_fg_raw[[order_col, family_col, fg_col]].copy()
    df_fg.columns = ["order", "family", "fg_id"]
    df_fg["order"] = df_fg["order"].astype(str).str.strip()
    df_fg["family"] = df_fg["family"].astype(str).str.strip()
    df_fg["fg_id"] = pd.to_numeric(df_fg["fg_id"], errors="coerce")
    df_fg = df_fg.dropna(subset=["fg_id"])
    df_fg["fg_id"] = df_fg["fg_id"].astype(int)
    df_fg = df_fg[(df_fg["family"].ne("")) & (df_fg["family"].ne("nan"))]

    mat = pd.read_excel(xlsx_path, sheet_name=matrix_sheet, header=None)
    raw = mat.values
    predator_ids = []
    pred_col_idx = []
    for i, v in enumerate(raw[0, 1:]):
        try:
            predator_ids.append(int(v))
            pred_col_idx.append(i + 1)
        except Exception:
            pass

    fg_edges: set[tuple[int, int]] = set()
    for r in range(1, len(raw)):
        row = raw[r]
        try:
            prey_id = int(row[0])
        except Exception:
            continue
        for i, col_idx in enumerate(pred_col_idx):
            predator_id = predator_ids[i]
            try:
                val = int(row[col_idx])
            except Exception:
                val = 0
            if val == 1:
                fg_edges.add((predator_id, prey_id))

    return df_fg.reset_index(drop=True), fg_edges


def load_layer_probabilities(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    family_col = "family_unit" if "family_unit" in df.columns else "taxon"
    out = df[["s3_sheet", "layer_seq", "layer_id", "suitability_probability", family_col]].copy()
    out = out.rename(columns={family_col: "family"})
    out["layer_seq"] = pd.to_numeric(out["layer_seq"], errors="coerce")
    out["suitability_probability"] = pd.to_numeric(out["suitability_probability"], errors="coerce")
    out = out.dropna(subset=["s3_sheet", "layer_seq", "layer_id", "family", "suitability_probability"])
    out["family"] = out["family"].astype(str).str.strip()
    out = out[out["family"].ne("")]
    return out


def build_family_graph(
    fg_map: pd.DataFrame,
    fg_edges: set[tuple[int, int]],
    modeled_families: set[str],
) -> tuple[set[str], set[tuple[str, str]], dict[str, str]]:
    fg_to_families: dict[int, list[str]] = {}
    family_to_order: dict[str, str] = {}
    for _, row in fg_map.iterrows():
        fam = str(row["family"]).strip()
        if fam not in modeled_families:
            continue
        fg = int(row["fg_id"])
        fg_to_families.setdefault(fg, []).append(fam)
        family_to_order[fam] = str(row["order"]).strip()

    baseline_nodes = set(family_to_order.keys())
    baseline_edges: set[tuple[str, str]] = set()
    for pred_fg, prey_fg in fg_edges:
        pred_list = fg_to_families.get(pred_fg, [])
        prey_list = fg_to_families.get(prey_fg, [])
        if not pred_list or not prey_list:
            continue
        for p in pred_list:
            for q in prey_list:
                if p == q:
                    continue
                baseline_edges.add((p, q))
    return baseline_nodes, baseline_edges, family_to_order


def build_adjacency(nodes: set[str], edges: set[tuple[str, str]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    prey_map = {n: set() for n in nodes}
    pred_map = {n: set() for n in nodes}
    for u, v in edges:
        if u in nodes and v in nodes:
            prey_map[u].add(v)
            pred_map[v].add(u)
    return prey_map, pred_map


def cascade_secondary_extinction(
    alive_initial: set[str],
    prey_map: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    alive = set(alive_initial)
    secondary_extinct: set[str] = set()
    while True:
        doomed = set()
        for fam in list(alive):
            prey_all = prey_map.get(fam, set())
            if not prey_all:
                continue
            active_prey = prey_all & alive
            if len(active_prey) == 0:
                doomed.add(fam)
        if not doomed:
            break
        alive -= doomed
        secondary_extinct |= doomed
    return alive, secondary_extinct


def edge_set_for_alive(alive: set[str], baseline_edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return {(u, v) for (u, v) in baseline_edges if u in alive and v in alive}


def jaccard_similarity(a: set[tuple[str, str]], b: set[tuple[str, str]]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def basin_sort_key(name: str) -> tuple[int, str]:
    m = re.match(r"^\s*(\d+)\s*[-_]", str(name))
    if m:
        return int(m.group(1)), str(name)
    return 10**9, str(name)


def basin_label_map(basins: list[str]) -> dict[str, str]:
    return {b: f"P{i+1}" for i, b in enumerate(sorted(basins, key=basin_sort_key))}


def safe_file_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text))


def evaluate_layers(
    layer_prob: pd.DataFrame,
    baseline_nodes: set[str],
    baseline_edges: set[tuple[str, str]],
) -> tuple[pd.DataFrame, dict[tuple[str, int], set[tuple[str, str]]]]:
    prey_map, _ = build_adjacency(baseline_nodes, baseline_edges)
    rows: list[dict[str, int | float | str]] = []
    link_sets: dict[tuple[str, int], set[tuple[str, str]]] = {}

    grouped = layer_prob.groupby(["s3_sheet", "layer_seq", "layer_id"], sort=False)
    for (basin, seq, layer_id), part in grouped:
        fam_prob = (
            part.groupby("family", as_index=False)["suitability_probability"].max().set_index("family")["suitability_probability"]
        )
        primary_alive = {f for f in baseline_nodes if fam_prob.get(f, 0.0) >= THRESHOLD_SURVIVE}
        primary_extinct = baseline_nodes - primary_alive

        final_alive, secondary_extinct = cascade_secondary_extinction(primary_alive, prey_map)
        final_links = edge_set_for_alive(final_alive, baseline_edges)

        predator_count = sum(1 for f in final_alive if len(prey_map.get(f, set()) & final_alive) > 0)
        link_count = len(final_links)
        jaccard_to_baseline = jaccard_similarity(final_links, baseline_edges)
        sec_rate = len(secondary_extinct) / len(baseline_nodes) if baseline_nodes else np.nan

        rows.append(
            {
                "basin": str(basin),
                "layer_seq": int(seq),
                "layer_id": str(layer_id),
                "primary_alive_count": int(len(primary_alive)),
                "primary_extinct_count": int(len(primary_extinct)),
                "secondary_extinct_count": int(len(secondary_extinct)),
                "secondary_extinction_rate": float(sec_rate),
                "predator_count": int(predator_count),
                "trophic_link_count": int(link_count),
                "jaccard_to_baseline": float(jaccard_to_baseline),
            }
        )
        link_sets[(str(basin), int(seq))] = final_links

    metrics_df = pd.DataFrame(rows).sort_values(["basin", "layer_seq"], key=lambda s: s.map(basin_sort_key) if s.name == "basin" else s, kind="stable").reset_index(drop=True)
    return metrics_df, link_sets


def internal_distance_by_basin(link_sets: dict[tuple[str, int], set[tuple[str, str]]]) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    basins = sorted({b for b, _ in link_sets.keys()}, key=basin_sort_key)
    for basin in basins:
        seqs = sorted([seq for b, seq in link_sets.keys() if b == basin])
        if not seqs:
            continue
        e0 = link_sets[(basin, seqs[0])]
        e_prev = e0
        for i, seq in enumerate(seqs):
            e_cur = link_sets[(basin, seq)]
            d_first = 1.0 - jaccard_similarity(e_cur, e0)
            d_prev = np.nan if i == 0 else (1.0 - jaccard_similarity(e_cur, e_prev))
            rows.append(
                {
                    "basin": basin,
                    "layer_seq": int(seq),
                    "distance_to_first": float(d_first),
                    "distance_to_previous": float(d_prev) if not np.isnan(d_prev) else np.nan,
                }
            )
            e_prev = e_cur
    return (
        pd.DataFrame(rows)
        .sort_values(["basin", "layer_seq"], key=lambda s: s.map(basin_sort_key) if s.name == "basin" else s, kind="stable")
        .reset_index(drop=True)
    )


def internal_trend_by_basin(internal_df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for basin in sorted(internal_df["basin"].astype(str).unique().tolist(), key=basin_sort_key):
        sub = internal_df[internal_df["basin"].astype(str) == basin].dropna(subset=["layer_seq", col])
        if len(sub) < 3:
            rows.append({"basin": basin, "metric": col, "spearman_rho": np.nan, "p_value": np.nan, "trend": "数据不足"})
            continue
        rho, p = spearmanr(sub["layer_seq"], sub[col])
        if np.isnan(rho) or np.isnan(p):
            trend = "趋势不可判定"
        elif p < 0.05 and rho > 0:
            trend = "盆地内部食物网差异随层位序号显著增大"
        elif p < 0.05 and rho < 0:
            trend = "盆地内部食物网差异随层位序号显著缩小"
        else:
            trend = "盆地内部食物网差异无显著单调变化"
        rows.append({"basin": basin, "metric": col, "spearman_rho": rho, "p_value": p, "trend": trend})
    return pd.DataFrame(rows)


def plot_metrics(metrics_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    basin_list = sorted(metrics_df["basin"].astype(str).unique().tolist(), key=basin_sort_key)
    metrics = [
        ("predator_count", "总捕食者数"),
        ("trophic_link_count", "营养链数量"),
        ("jaccard_to_baseline", "营养链组成Jaccard"),
        ("secondary_extinction_rate", "次级灭绝率"),
    ]
    cmap = plt.get_cmap("tab10")
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for i, basin in enumerate(basin_list):
            sub = metrics_df[metrics_df["basin"].astype(str) == basin].sort_values("layer_seq")
            ax.plot(sub["layer_seq"], sub[col], marker="o", ms=2.5, lw=1.2, color=cmap(i % 10), label=basin)
        ax.set_title(title)
        ax.set_xlabel("layer_seq")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    axes[0, 0].set_ylabel("value")
    axes[1, 0].set_ylabel("value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_single_basin_by_layerseq(
    metrics_df: pd.DataFrame,
    metric_col: str,
    title_prefix: str,
    y_label: str,
    out_dir: Path,
) -> list[Path]:
    out_paths: list[Path] = []
    basins = sorted(metrics_df["basin"].astype(str).unique().tolist(), key=basin_sort_key)
    p_labels = basin_label_map(basins)
    cmap = plt.get_cmap("tab10")
    for i, basin in enumerate(basins):
        sub = metrics_df[metrics_df["basin"].astype(str) == basin].sort_values("layer_seq")
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        x = sub["layer_seq"].to_numpy()
        y = sub[metric_col].to_numpy()
        ax.plot(x, y, marker="o", ms=3.0, lw=1.5, color=cmap(i % 10))
        ax.set_xlabel("layer_seq（按No列顺序）")
        ax.set_ylabel(y_label)
        ax.set_title(f"{title_prefix} - {p_labels[basin]} ({basin})")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        fig.tight_layout()
        out_path = out_dir / f"{metric_col}_{safe_file_name(basin)}.pdf"
        fig.savefig(out_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def plot_internal_distance(internal_df: pd.DataFrame, col: str, out_path: Path, title: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    basins = sorted(internal_df["basin"].astype(str).unique().tolist(), key=basin_sort_key)
    cmap = plt.get_cmap("tab10")
    for i, basin in enumerate(basins):
        sub = internal_df[internal_df["basin"].astype(str) == basin].sort_values("layer_seq")
        ax.plot(sub["layer_seq"], sub[col], marker="o", ms=2.5, lw=1.2, color=cmap(i % 10), label=basin)
    ax.set_title(title)
    ax.set_xlabel("layer_seq")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    paths = detect_paths(work_dir)

    fg_map, fg_edges = load_foodweb_reference(paths.ref_foodweb_xlsx)
    layer_prob = load_layer_probabilities(paths.maxent_long_csv)
    modeled_families = set(layer_prob["family"].astype(str).unique().tolist())

    baseline_nodes, baseline_edges, family_to_order = build_family_graph(fg_map, fg_edges, modeled_families)
    metrics_df, link_sets = evaluate_layers(layer_prob, baseline_nodes, baseline_edges)

    basin_summary_df = (
        metrics_df.groupby("basin", as_index=False)[
            ["predator_count", "trophic_link_count", "jaccard_to_baseline", "secondary_extinction_rate"]
        ].mean()
    )

    internal_df = internal_distance_by_basin(link_sets)
    trend_first_df = internal_trend_by_basin(internal_df, "distance_to_first")
    trend_prev_df = internal_trend_by_basin(internal_df, "distance_to_previous")

    fg_map.to_csv(paths.out_dir / "ceg_functional_group_mapping.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sorted(list(fg_edges)), columns=["predator_fg", "prey_fg"]).to_csv(
        paths.out_dir / "ceg_fg_edges.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(sorted(list(baseline_nodes)), columns=["family"]).to_csv(
        paths.out_dir / "foodweb_family_nodes.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(sorted(list(baseline_edges)), columns=["predator_family", "prey_family"]).to_csv(
        paths.out_dir / "foodweb_family_edges.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [{"family": f, "order": family_to_order.get(f, "")} for f in sorted(baseline_nodes)]
    ).to_csv(paths.out_dir / "foodweb_family_order_mapping.csv", index=False, encoding="utf-8-sig")

    metrics_df.to_csv(paths.out_dir / "layer_foodweb_metrics.csv", index=False, encoding="utf-8-sig")
    basin_summary_df.to_csv(paths.out_dir / "basin_foodweb_metrics_mean.csv", index=False, encoding="utf-8-sig")
    internal_df.to_csv(paths.out_dir / "basin_internal_foodweb_distance_by_layer.csv", index=False, encoding="utf-8-sig")
    trend_first_df.to_csv(paths.out_dir / "basin_internal_foodweb_trend_to_first.csv", index=False, encoding="utf-8-sig")
    trend_prev_df.to_csv(paths.out_dir / "basin_internal_foodweb_trend_to_previous.csv", index=False, encoding="utf-8-sig")

    plot_metrics(metrics_df, paths.fig_dir / "layer_foodweb_metrics_by_basin.pdf")
    per_basin_dir = paths.fig_dir / "per_basin_metrics"
    per_basin_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_single_basin_by_layerseq(
        metrics_df, "predator_count", "总捕食者数按层位变化", "predator_count", per_basin_dir
    )
    plot_metric_single_basin_by_layerseq(
        metrics_df, "trophic_link_count", "营养链数量按层位变化", "trophic_link_count", per_basin_dir
    )
    plot_metric_single_basin_by_layerseq(
        metrics_df, "jaccard_to_baseline", "营养链组成Jaccard按层位变化", "jaccard_to_baseline", per_basin_dir
    )
    plot_metric_single_basin_by_layerseq(
        metrics_df,
        "secondary_extinction_rate",
        "次级灭绝率按层位变化",
        "secondary_extinction_rate",
        per_basin_dir,
    )
    plot_internal_distance(
        internal_df,
        "distance_to_first",
        paths.fig_dir / "basin_internal_foodweb_distance_to_first_trend.pdf",
        "六盆地内部食物网差异随层位变化（相对本盆地首层）",
        "distance to basin first layer",
    )
    plot_internal_distance(
        internal_df,
        "distance_to_previous",
        paths.fig_dir / "basin_internal_foodweb_distance_to_previous_trend.pdf",
        "六盆地内部食物网差异随层位变化（相邻层位）",
        "distance to previous layer",
    )

    summary_lines = [
        "# 5.3 CEG食物网分析结果摘要",
        "",
        f"- 参考FG数量: {fg_map['fg_id'].nunique()}",
        f"- 参考FG捕食关系数: {len(fg_edges)}",
        f"- 纳入分析的family节点数: {len(baseline_nodes)}",
        f"- family营养链数量(基准): {len(baseline_edges)}",
        f"- 分层位指标记录数: {len(metrics_df)}",
        f"- 盆地内部差异趋势(to_first)记录数: {len(trend_first_df)}",
        f"- 盆地内部差异趋势(to_previous)记录数: {len(trend_prev_df)}",
        "",
        "## 指标说明",
        f"- 初级灭绝阈值: MaxEnt 存活概率 < {THRESHOLD_SURVIVE}",
        "- 次级灭绝规则: 捕食者若失去全部猎物则灭绝，直至级联稳定。",
        "- 营养链组成Jaccard: 与基准营养链集合相比的Jaccard相似度。",
        "- 盆地内部差异量化: 各盆地层位链集合相对首层的Jaccard距离，并计算相邻层位距离。",
        "",
        "## 输出文件",
        "- layer_foodweb_metrics.csv",
        "- basin_foodweb_metrics_mean.csv",
        "- basin_internal_foodweb_distance_by_layer.csv",
        "- basin_internal_foodweb_trend_to_first.csv",
        "- basin_internal_foodweb_trend_to_previous.csv",
        "- figures/layer_foodweb_metrics_by_basin.pdf",
        "- figures/per_basin_metrics/*.pdf（每盆地每指标单图）",
        "- figures/basin_internal_foodweb_distance_to_first_trend.pdf",
        "- figures/basin_internal_foodweb_distance_to_previous_trend.pdf",
    ]
    (paths.out_dir / "summary_report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("5.3 CEG食物网分析完成。")
    print(f"结果目录: {paths.out_dir}")


if __name__ == "__main__":
    main()


