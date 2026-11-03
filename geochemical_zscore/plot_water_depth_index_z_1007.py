from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


INDICATOR_KEY = "water_depth_index_z"
INDICATOR_NAME = "水深指标"

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def find_directory_by_codepoint(parent: Path, codepoint: int) -> Path:
    matches = [
        path
        for path in parent.iterdir()
        if path.is_dir() and any(ord(char) == codepoint for char in path.name)
    ]
    if not matches:
        raise FileNotFoundError(f"未找到目标目录：U+{codepoint:04X}")
    return matches[0]


def select_input_files(code_dir: Path) -> tuple[Path, Path]:
    upload_root = code_dir.resolve().parents[1]
    data_dir = upload_root / "data"
    sdata2_path = data_dir / "dataset_002.xlsx"
    sdata3_path = data_dir / "dataset_005.xlsx"
    missing = [str(path) for path in (sdata2_path, sdata3_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少上传包数据文件: {missing}")
    return sdata2_path, sdata3_path


def build_points(profile_by_sheet: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for basin, profile in profile_by_sheet.items():
        if INDICATOR_KEY not in profile.columns:
            continue
        part = profile[["layer_seq", "layer_id", INDICATOR_KEY]].copy()
        part = part.rename(columns={INDICATOR_KEY: "indicator_value"})
        part.insert(0, "basin", str(basin))
        rows.append(part)
    if not rows:
        raise ValueError("1007层位数据中未生成水深Z-score指标。")
    return pd.concat(rows, ignore_index=True)


def plot_profile(points: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 11))
    colors = plt.get_cmap("tab10")

    for index, (basin, group) in enumerate(points.groupby("basin", sort=False)):
        group = group.sort_values("layer_seq")
        axis.plot(
            group["indicator_value"],
            group["layer_seq"],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            color=colors(index % 10),
            label=basin,
            alpha=0.95,
        )

    axis.axvline(0.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.8)
    axis.invert_yaxis()
    axis.set_xlabel("综合得分（Z-score 均值）")
    axis.set_ylabel("样本序号（按1007版原始层位自上到下）")
    axis.set_title("水深指标：多盆地纵向折线图")
    max_seq = int(points["layer_seq"].max())
    tick_step = 5 if max_seq >= 20 else 2
    axis.set_yticks(range(1, max_seq + 1, tick_step))
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    axis.legend(loc="lower right", fontsize=9, frameon=True)
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    if str(task_dir) not in sys.path:
        sys.path.insert(0, str(task_dir))

    from maxent_species_prediction import (
        loadEnvTrainingData,
        loadProfileDataBySheet,
    )

    sdata2_path, sdata3_path = select_input_files(task_dir)
    _, z_stats = loadEnvTrainingData(sdata2_path)
    profile_by_sheet = loadProfileDataBySheet(sdata3_path, z_stats)
    points = build_points(profile_by_sheet)

    figure_dir = task_dir / "figures"
    table_dir = task_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_profile(points, figure_dir / "fig_07_profile_water_depth_index_z.pdf")
    points.to_csv(
        table_dir / "fig_07_profile_water_depth_index_z_points.csv",
        index=False,
        encoding="utf-8-sig",
    )

    methods = f"""# 水深 Z-score 图方法说明

- 剖面输入：`{sdata3_path.name}`。
- Z-score基准：生态六 S data-2 环境训练数据的均值和总体标准差。
- 水深指标：`Fe/Mn` 与 `—(Al+Fe)/(Ca+Mg)` 两项标准化值的均值。
- 层位顺序：1007版 S data-3 各sheet中的原始行顺序。
- 图形样式：参考生态五 `fig_07_profile_water_depth_index_z`，横轴为水深Z-score，纵轴为层位序号并反向显示。
- 图件同时输出PDF和PNG；逐点数值输出CSV。
"""
    (task_dir / "methods.md").write_text(methods, encoding="utf-8")
    print(f"完成：{figure_dir / 'fig_07_profile_water_depth_index_z.pdf'}")
    print(f"数据点：{len(points)}，盆地数：{points['basin'].nunique()}")


if __name__ == "__main__":
    main()
