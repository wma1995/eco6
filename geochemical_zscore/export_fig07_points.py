from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

import pandas as pd


INDICATOR_META: Dict[str, str] = {
    "salinity_index_z": "盐度指标",
    "temperature_index_raw": "温度指标",
    "water_depth_index_z": "水深指标",
    "humidity_index_z": "湿度指标",
    "volcanic_intensity_index_z": "火山喷发强度指标",
    "redox_index_z": "水体氧化势指标",
}


def locate_output_data_dir(base_dir: Path) -> Path:
    output_dir = base_dir / "results" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))
    from maxent_species_prediction import detectFiles, loadEnvTrainingData, loadProfileDataBySheet

    paths = detectFiles(base_dir)
    output_data_dir = locate_output_data_dir(base_dir)

    _, z_stats = loadEnvTrainingData(paths.sdata2)
    profile_by_sheet = loadProfileDataBySheet(paths.sdata3, z_stats)

    rows = []
    for indicator_key, indicator_cn in INDICATOR_META.items():
        fig_name = f"fig_07_profile_{indicator_key}"
        for basin, df in profile_by_sheet.items():
            if indicator_key not in df.columns:
                continue
            work = df[["layer_seq", "layer_id", indicator_key]].copy()
            work = work.rename(columns={indicator_key: "indicator_value"})
            work["fig_name"] = fig_name
            work["indicator_key"] = indicator_key
            work["indicator_cn"] = indicator_cn
            work["basin"] = str(basin)
            rows.append(work)

    if not rows:
        raise ValueError("未提取到 fig_07 指标点。")

    out_df = pd.concat(rows, ignore_index=True)
    out_df = out_df[
        ["fig_name", "indicator_key", "indicator_cn", "basin", "layer_seq", "layer_id", "indicator_value"]
    ].sort_values(["indicator_key", "basin", "layer_seq"], kind="stable")

    long_csv = output_data_dir / "fig_07_series_indicator_points_long.csv"
    out_df.to_csv(long_csv, index=False, encoding="utf-8-sig")

                                  
    for indicator_key in INDICATOR_META:
        part = out_df[out_df["indicator_key"] == indicator_key].copy()
        part_csv = output_data_dir / f"fig_07_profile_{indicator_key}_points.csv"
        part.to_csv(part_csv, index=False, encoding="utf-8-sig")

    print("fig_07 系列指标点表格导出完成。")
    print(f"输出目录: {output_data_dir}")
    print(f"总记录数: {len(out_df)}")


if __name__ == "__main__":
    main()
