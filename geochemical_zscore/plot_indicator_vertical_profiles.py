from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from maxent_species_prediction import (
    detectFiles,
    loadEnvTrainingData,
    loadProfileDataBySheet,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


INDICATOR_META: Dict[str, Dict[str, str | bool]] = {
    "salinity_index_z": {"name_cn": "盐度指标", "is_zscore": True},
    "temperature_index_raw": {"name_cn": "温度指标", "is_zscore": False},
    "water_depth_index_z": {"name_cn": "水深指标", "is_zscore": True},
    "humidity_index_z": {"name_cn": "湿度指标", "is_zscore": True},
    "volcanic_intensity_index_z": {"name_cn": "火山喷发强度指标", "is_zscore": True},
    "redox_index_z": {"name_cn": "水体氧化势指标", "is_zscore": True},
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_sheet_display_name(sheet_name: str) -> str:
    return str(sheet_name)


def plot_single_indicator(
    indicator_key: str,
    profile_by_sheet: Dict[str, object],
    out_dir: Path,
) -> Path:
    meta = INDICATOR_META[indicator_key]
    name_cn = str(meta["name_cn"])
    is_zscore = bool(meta["is_zscore"])

    plt.figure(figsize=(8.5, 11))
    cmap = plt.get_cmap("tab10")

    max_seq = 0
    for i, (sheet_name, df) in enumerate(profile_by_sheet.items()):
        if indicator_key not in df.columns:
            continue
        y = df["layer_seq"]
        x = df[indicator_key]
        max_seq = max(max_seq, int(y.max()))
        plt.plot(
            x,
            y,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            color=cmap(i % 10),
            label=make_sheet_display_name(sheet_name),
            alpha=0.95,
        )

    plt.gca().invert_yaxis()
    if is_zscore:
        plt.axvline(0.0, linestyle="--", linewidth=1.0, color="gray", alpha=0.8)
        plt.xlabel("综合得分（Z-score 均值）")
    else:
        plt.xlabel("指标值（原值）")
    plt.ylabel("样本序号（按原始层位自上到下）")
    plt.title(f"{name_cn}：多钻井样本纵向折线图")

    if max_seq > 0:
        tick_step = 5 if max_seq >= 20 else 2
        ticks = list(range(1, max_seq + 1, tick_step))
        plt.yticks(ticks)
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()

    out_pdf = out_dir / f"fig_07_profile_{indicator_key}.pdf"
    out_png = out_dir / f"fig_07_profile_{indicator_key}.png"
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    return out_pdf


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    paths = detectFiles(base_dir)
    out_dir = paths.output_dir / "figures_profiles"
    ensure_dir(out_dir)

    _, z_stats = loadEnvTrainingData(paths.sdata2)
    profile_by_sheet = loadProfileDataBySheet(paths.sdata3, z_stats)

    generated = []
    for indicator_key in INDICATOR_META:
        generated.append(plot_single_indicator(indicator_key, profile_by_sheet, out_dir))

    print("纵向折线图生成完成。")
    print(f"输出目录: {out_dir}")
    print(f"生成数量: {len(generated)}")


if __name__ == "__main__":
    main()

