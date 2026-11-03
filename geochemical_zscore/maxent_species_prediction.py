from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

MIN_PRESENCE = 3
RANDOM_STATE = 42
TEST_SIZE = 0.3

INDICATOR_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "salinity_index_z": {
        "components": ["(Mg/Al2O3)*100", "Sr/Ba", "Mg/Ca", "B/Ga"],
        "use_zscore": True,
    },
    "temperature_index_raw": {
        "components": ["CaO/(MgO·Al2O3)"],
        "use_zscore": False,
    },
    "water_depth_index_z": {
        "components": ["Fe/Mn", "—(Al+Fe)/(Ca+Mg)"],
        "use_zscore": True,
    },
    "humidity_index_z": {
        "components": ["CIA", "Al2O3/SiO2", "CIX"],
        "use_zscore": True,
    },
    "volcanic_intensity_index_z": {
        "components": ["-Ni/Al2O3", "-Cr/Al2O3"],
        "use_zscore": True,
    },
    "redox_index_z": {
        "components": ["Ni/Co", "Cu/Zn"],
        "use_zscore": True,
    },
}


@dataclass
class TPaths:
    sdata1: Path
    sdata2: Path
    sdata3: Path
    sdata4: Path
    output_dir: Path


def normalizeKey(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def canonicalFormula(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def hasNegativePrefix(text: str) -> bool:
    return bool(re.match(r"^\s*[-—－]", str(text)))


def chooseFamilyUnit(row: pd.Series) -> str:
    val = str(row.get("family", "")).strip()
    invalid = {"", "nan", "unknown", "unknown_family", "unknown family"}
    if val and val.lower() not in invalid:
        return val
    return ""


def cleanAliasHeader(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(axis=1, how="all").dropna(axis=0, how="all").copy()
    if cleaned.empty:
        return cleaned
    first_row = cleaned.iloc[0].astype(str).str.strip()
    if first_row.str.contains(r"^No\.$", case=False, regex=True).any():
        new_cols: List[str] = []
        for col, alias in zip(cleaned.columns, first_row):
            alias = alias.strip()
            new_cols.append(alias if alias and alias.lower() != "nan" else str(col))
        cleaned.columns = new_cols
        cleaned = cleaned.iloc[1:].copy()
    return cleaned


def detectFiles(base_dir: Path) -> TPaths:
    upload_root = Path(base_dir).resolve().parent
    data_dir = upload_root / "data"
    required = {
        "sdata1": data_dir / "dataset_001.xlsx",
        "sdata2": data_dir / "dataset_002.xlsx",
        "sdata3": data_dir / "dataset_003.xlsx",
        "sdata4": data_dir / "dataset_004.xlsx",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少上传包数据文件: {missing}")
    return TPaths(
        sdata1=required["sdata1"],
        sdata2=required["sdata2"],
        sdata3=required["sdata3"],
        sdata4=required["sdata4"],
        output_dir=Path(base_dir).resolve() / "results",
    )

def loadFoodWebMapping(sdata4_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(sdata4_path, sheet_name="Functional group")
    except Exception:
        return pd.DataFrame(columns=["order", "family", "functional_group"])
    needed = [c for c in df.columns if str(c).lower() in {"order", "family", "functional group"}]
    if len(needed) < 3:
        return pd.DataFrame(columns=["order", "family", "functional_group"])
    out = df[needed].copy()
    out.columns = ["order", "family", "functional_group"]
    out = out.dropna(how="all")
    out["order"] = out["order"].astype(str).str.strip()
    out["family"] = out["family"].astype(str).str.strip()
    out["functional_group"] = out["functional_group"].astype(str).str.strip()
    return out


def _extractByCanonical(raw_df: pd.DataFrame, canonical_key: str) -> Tuple[pd.Series | None, str | None]:
    for col in raw_df.columns:
        if canonicalFormula(col) == canonical_key:
            ser = pd.to_numeric(raw_df[col], errors="coerce")
            return ser, str(col)
    return None, None


def extractComponentSeries(raw_df: pd.DataFrame, component: str) -> pd.Series | None:
    comp = component.strip()

    if comp == "(Mg/Al2O3)*100":
        ser, _ = _extractByCanonical(raw_df, "mgal2o3100")
        return ser
    if comp == "Sr/Ba":
        ser, _ = _extractByCanonical(raw_df, "srba")
        return ser
    if comp == "Mg/Ca":
        ser, _ = _extractByCanonical(raw_df, "mgca")
        return ser
    if comp == "B/Ga":
        ser, _ = _extractByCanonical(raw_df, "bga")
        return ser
    if comp == "CaO/(MgO·Al2O3)":
        ser, _ = _extractByCanonical(raw_df, "caomgoal2o3")
        return ser
    if comp == "Fe/Mn":
        ser, _ = _extractByCanonical(raw_df, "femn")
        return ser
    if comp == "—(Al+Fe)/(Ca+Mg)":
        ser, _ = _extractByCanonical(raw_df, "alfecamg")
        return ser
    if comp == "CIA":
        ser, _ = _extractByCanonical(raw_df, "cia")
        return ser
    if comp == "CIX":
        ser, _ = _extractByCanonical(raw_df, "cix")
        return ser
    if comp == "Al2O3/SiO2":
        ser, _ = _extractByCanonical(raw_df, "al2o3sio2")
        if ser is not None:
            return ser
        reciprocal, _ = _extractByCanonical(raw_df, "sio2al2o3")
        if reciprocal is None:
            return None
        out = 1.0 / reciprocal.replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    if comp == "-Ni/Al2O3":
        ser, col = _extractByCanonical(raw_df, "nial2o3")
        if ser is None:
            return None
        if col is not None and not hasNegativePrefix(col):
            return -ser
        return ser

    if comp == "-Cr/Al2O3":
        ser, col = _extractByCanonical(raw_df, "cral2o3")
        if ser is None:
            return None
        if col is not None and not hasNegativePrefix(col):
            return -ser
        return ser

    if comp == "Ni/Co":
        ser, col = _extractByCanonical(raw_df, "nico")
        if ser is None:
            return None
                                                            
        if col is not None and hasNegativePrefix(col):
            return -ser
        return ser

    if comp == "Cu/Zn":
        ser, _ = _extractByCanonical(raw_df, "cuzn")
        return ser

    return None


def buildIndicatorFrame(
    raw_df: pd.DataFrame,
    z_stats: Dict[str, Tuple[float, float]] | None = None,
    fit_stats: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {} if z_stats is None else dict(z_stats)
    component_values: Dict[str, pd.Series] = {}
    z_components: List[str] = []

    for rule in INDICATOR_DEFINITIONS.values():
        components = rule["components"]                       
        use_zscore = bool(rule["use_zscore"])
        for comp in components:
            comp = str(comp)
            if comp in component_values:
                continue
            ser = extractComponentSeries(raw_df, comp)
            if ser is not None:
                component_values[comp] = ser
            if use_zscore and comp not in z_components:
                z_components.append(comp)

    if fit_stats:
        for comp in z_components:
            ser = component_values.get(comp)
            if ser is None:
                stats[comp] = (0.0, 1.0)
                continue
            mu = float(ser.mean(skipna=True))
            sd = float(ser.std(skipna=True, ddof=0))
            if not np.isfinite(sd) or sd == 0.0:
                sd = 1.0
            if not np.isfinite(mu):
                mu = 0.0
            stats[comp] = (mu, sd)

    out = pd.DataFrame(index=raw_df.index)
    for indicator_name, rule in INDICATOR_DEFINITIONS.items():
        components = [str(c) for c in rule["components"]]                       
        use_zscore = bool(rule["use_zscore"])

        if use_zscore:
            values = []
            for comp in components:
                ser = component_values.get(comp)
                if ser is None:
                    continue
                mu, sd = stats.get(comp, (0.0, 1.0))
                if sd == 0.0 or not np.isfinite(sd):
                    sd = 1.0
                values.append((ser - mu) / sd)
            if values:
                out[indicator_name] = pd.concat(values, axis=1).mean(axis=1, skipna=True)
            else:
                out[indicator_name] = np.nan
        else:
            ser = component_values.get(components[0])
            out[indicator_name] = ser if ser is not None else np.nan

    return out.replace([np.inf, -np.inf], np.nan), stats


def loadEnvTrainingData(sdata2_path: Path) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    raw = pd.read_excel(sdata2_path, sheet_name=0)
    raw = cleanAliasHeader(raw)
    if raw.empty:
        raise ValueError("S data-2 为空，无法建模。")

    first_col = raw.columns[0]
    raw = raw.rename(columns={first_col: "site_id"})
    raw["site_id"] = raw["site_id"].astype(str).str.strip()
    raw = raw[raw["site_id"].ne("") & raw["site_id"].ne("nan")]

    indicators, z_stats = buildIndicatorFrame(raw, fit_stats=True)
    env_df = pd.concat([raw[["site_id"]].reset_index(drop=True), indicators.reset_index(drop=True)], axis=1)
    return env_df, z_stats


def buildSiteTaxaMatrix(sdata1_path: Path, env_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    env_site_exact = {str(s).strip(): str(s).strip() for s in env_df["site_id"]}
    env_site_lower = {str(s).strip().lower(): str(s).strip() for s in env_df["site_id"]}
    env_site_lookup = {normalizeKey(s): s for s in env_df["site_id"]}
    xls = pd.ExcelFile(sdata1_path)
    site_taxa_counts: Dict[str, Dict[str, int]] = {}
    taxa_meta: Dict[str, Dict[str, str]] = {}
    mapping_rows: List[Dict[str, object]] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(sdata1_path, sheet_name=sheet, usecols="A:F")
        df.columns = ["sample_number", "order", "family", "genera", "species", "taphonomic_grade"]
        df["sample_number"] = df["sample_number"].astype(str).str.strip()
        df = df[df["sample_number"].ne("") & df["sample_number"].ne("nan")]

        sheet_raw = str(sheet).strip()
        matched_site = env_site_exact.get(sheet_raw)
        if matched_site is None:
            matched_site = env_site_lower.get(sheet_raw.lower())
        if matched_site is None:
            sheet_key = normalizeKey(sheet_raw)
            matched_site = env_site_lookup.get(sheet_key)
        if matched_site is None:
            sheet_key = normalizeKey(sheet_raw)
            candidates = [site for key, site in env_site_lookup.items() if sheet_key in key or key in sheet_key]
            if len(candidates) == 1:
                matched_site = candidates[0]

        taxa_count = 0 if df.empty else int(
            df["family"]
            .astype(str)
            .str.strip()
            .replace("nan", "")
            .loc[lambda s: s.ne("")]
            .nunique()
        )
        mapping_rows.append(
            {
                "sdata1_sheet": sheet,
                "matched_site_id": matched_site if matched_site is not None else "",
                "record_count": int(len(df)),
                "taxa_count": taxa_count,
            }
        )
        if df.empty or matched_site is None:
            continue

        df["taxon"] = df.apply(chooseFamilyUnit, axis=1)
        df = df[df["taxon"].ne("")]
        if df.empty:
            continue

        counts = df["taxon"].value_counts()
        site_taxa_counts.setdefault(matched_site, {})
        for taxon, count in counts.items():
            site_taxa_counts[matched_site][taxon] = site_taxa_counts[matched_site].get(taxon, 0) + int(count)

        meta = df[["taxon", "order", "family"]].drop_duplicates("taxon")
        for _, row in meta.iterrows():
            taxa_meta.setdefault(
                row["taxon"],
                {
                    "order": row["order"],
                    "family": row["family"],
                },
            )

    abundance = pd.DataFrame(site_taxa_counts).T.fillna(0).astype(int)
    abundance.index.name = "site_id"
    meta_df = pd.DataFrame(taxa_meta).T.reset_index().rename(columns={"index": "taxon"})
    mapping_df = pd.DataFrame(mapping_rows)
    return abundance, meta_df, mapping_df


def loadProfileDataBySheet(
    sdata3_path: Path,
    z_stats: Dict[str, Tuple[float, float]],
) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(sdata3_path)
    profile_by_sheet: Dict[str, pd.DataFrame] = {}
    for sheet in xls.sheet_names:
        raw = pd.read_excel(sdata3_path, sheet_name=sheet)
        df = cleanAliasHeader(raw)
        if df.empty:
            continue
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").copy()
        if df.empty:
            continue

        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        id_col = "No." if "No." in df.columns else (object_cols[0] if object_cols else df.columns[0])
        df = df.rename(columns={id_col: "layer_id"})
        df["layer_id"] = df["layer_id"].astype(str).str.strip()
        df["layer_seq"] = np.arange(1, len(df) + 1)
        missing_mask = df["layer_id"].isin(["", "nan"])
        if missing_mask.any():
            df.loc[missing_mask, "layer_id"] = [
                f"{sheet}_{int(v)}" for v in df.loc[missing_mask, "layer_seq"].tolist()
            ]

        indicators, _ = buildIndicatorFrame(df, z_stats=z_stats, fit_stats=False)
        profile_df = pd.concat([df[["layer_seq", "layer_id"]].reset_index(drop=True), indicators.reset_index(drop=True)], axis=1)
        profile_by_sheet[sheet] = profile_df

    if not profile_by_sheet:
        raise ValueError("S data-3 中未读取到可预测剖面数据。")
    return profile_by_sheet


def trainAndPredict(
    env_df: pd.DataFrame,
    abundance_df: pd.DataFrame,
    profile_by_sheet: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged_sites = env_df.merge(abundance_df, left_on="site_id", right_index=True, how="inner")
    feature_cols = [c for c in env_df.columns if c != "site_id"]
    X = merged_sites[feature_cols].copy()
    taxa_cols = [c for c in merged_sites.columns if c not in ["site_id", *feature_cols]]

    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    feature_medians = X.median(numeric_only=True)

    prepared_profiles: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sheet_name, df in profile_by_sheet.items():
        profile_X = pd.DataFrame(index=df.index)
        for col in feature_cols:
            if col in df.columns:
                profile_X[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                profile_X[col] = np.nan
        profile_X = profile_X.fillna(feature_medians)
        profile_meta = df[["layer_seq", "layer_id"]].copy()
        profile_meta["s3_sheet"] = sheet_name
        prepared_profiles[sheet_name] = {"X": profile_X, "meta": profile_meta}

    metrics_rows, roc_rows, contrib_rows, pred_rows = [], [], [], []
    for taxon in taxa_cols:
        y = (merged_sites[taxon] > 0).astype(int)
        pos_count = int(y.sum())
        neg_count = int(len(y) - pos_count)
        if pos_count == 0 or neg_count == 0:
            metrics_rows.append(
                {
                    "taxon": taxon,
                    "presence_sites": pos_count,
                    "absence_sites": neg_count,
                    "auc": np.nan,
                    "n_features": len(feature_cols),
                    "model_status": "skipped_single_class",
                    "evaluation_mode": "none",
                }
            )
            continue

        model = Pipeline(
            steps=[
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
            ]
        )

        if min(pos_count, neg_count) >= MIN_PRESENCE:
            x_train, x_test, y_train, y_test = train_test_split(
                X[feature_cols], y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            model.fit(x_train, y_train)
            y_prob = model.predict_proba(x_test)[:, 1]
            evaluation_mode = "holdout_stratified"
        else:
            model.fit(X[feature_cols], y)
            y_test = y
            y_prob = model.predict_proba(X[feature_cols])[:, 1]
            evaluation_mode = "resubstitution_low_sample"

        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        metrics_rows.append(
            {
                "taxon": taxon,
                "presence_sites": pos_count,
                "absence_sites": neg_count,
                "auc": auc,
                "n_features": len(feature_cols),
                "model_status": "ok",
                "evaluation_mode": evaluation_mode,
            }
        )
        for fp, tp, th in zip(fpr, tpr, thresholds):
            roc_rows.append({"taxon": taxon, "fpr": fp, "tpr": tp, "threshold": th})

        model.fit(X[feature_cols], y)
        coef = model.named_steps["clf"].coef_[0]
        abs_coef = np.abs(coef)
        contribution = abs_coef / abs_coef.sum() * 100 if abs_coef.sum() > 0 else np.zeros_like(abs_coef)
        for feat, val in zip(feature_cols, contribution):
            contrib_rows.append({"taxon": taxon, "feature": feat, "contribution_percent": float(val)})

        for sheet_name, sheet_pack in prepared_profiles.items():
            layer_prob = model.predict_proba(sheet_pack["X"][feature_cols])[:, 1]
            for i, p in enumerate(layer_prob):
                pred_rows.append(
                    {
                        "s3_sheet": sheet_name,
                        "layer_seq": int(sheet_pack["meta"].iloc[i]["layer_seq"]),
                        "layer_id": sheet_pack["meta"].iloc[i]["layer_id"],
                        "taxon": taxon,
                        "suitability_probability": float(p),
                    }
                )

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(roc_rows),
        pd.DataFrame(contrib_rows),
        pd.DataFrame(pred_rows),
    )


def writeMethodDescription(
    output_dir: Path,
    feature_count: int,
    model_count: int,
    sdata1_sheet_count: int,
    sdata3_sheet_count: int,
) -> None:
    text = f"""# MaxEnt模型参数与方法说明（论文可用）

## 1. 建模目标
基于 `S data-1`（样方化石记录）与 `S data-2`（样方地球化学环境因子）构建最大熵模型，
预测科级分类单元在 `S data-3` 不同层位中的适生概率。

## 2. 指标构建与标准化
原始环境变量按研究设定聚合为 6 个指标：
1. 盐度指标：Z[(Mg/Al2O3)*100, Sr/Ba, Mg/Ca, B/Ga]
2. 温度指标：CaO/(MgO·Al2O3)（单变量，按原值使用）
3. 水深指标：Z[Fe/Mn, —(Al+Fe)/(Ca+Mg)]
4. 湿度指标：Z[CIA, Al2O3/SiO2, CIX]
5. 火山喷发强度指标：Z[-Ni/Al2O3, -Cr/Al2O3]
6. 水体氧化势指标：Z[Ni/Co, Cu/Zn]

其中 Z 表示对各分量先做 z-score，再对同组分量取均值。

## 3. 样本定义与模型参数
- `S data-1` 每个sheet代表一个样方，共 {sdata1_sheet_count} 个样方；
- 以 family 为建模单元，出现/未出现编码为 1/0；
- 分类器：`LogisticRegression`（MaxEnt 对数线性等价实现）
- 求解器：`liblinear`
- 类别权重：`balanced`
- 最大迭代：`2000`
- 留出测试比例：`{TEST_SIZE}`
- 随机种子：`{RANDOM_STATE}`

## 4. 评估与解释
- 性能指标：AUC 与 ROC
- 样本充足时采用分层留出评估；低样本时采用全样本评估（确保 ROC 可计算）
- 变量贡献率：`|beta_i| / sum_j(|beta_j|) * 100%`

## 5. 层位预测
`S data-3` 共 {sdata3_sheet_count} 个sheet，各sheet独立计算后汇总；
每个sheet内层位顺序按原始行顺序（`layer_seq`）保留。

## 6. 结果输出
- 训练特征数：{feature_count}
- 成功建模科级单元数：{model_count}
- 输出包括：AUC、ROC曲线点、变量贡献率、分层位适生概率长表与宽表。
"""
    (output_dir / "method_description.md").write_text(text, encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    paths = detectFiles(base_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    env_df, z_stats = loadEnvTrainingData(paths.sdata2)
    abundance_df, taxa_meta_df, site_mapping_df = buildSiteTaxaMatrix(paths.sdata1, env_df)
    profile_by_sheet = loadProfileDataBySheet(paths.sdata3, z_stats)
    food_web_df = loadFoodWebMapping(paths.sdata4)

    metrics_df, roc_df, contrib_df, preds_long_df = trainAndPredict(env_df, abundance_df, profile_by_sheet)
    if metrics_df.empty:
        raise RuntimeError("未训练出可用模型，请检查样本量、特征列和最小出现阈值设置。")

    preds_wide_df = preds_long_df.pivot_table(
        index=["s3_sheet", "layer_seq", "layer_id"],
        columns="taxon",
        values="suitability_probability",
        aggfunc="first",
    ).reset_index()
    preds_wide_df = preds_wide_df.sort_values(["s3_sheet", "layer_seq"], kind="stable").reset_index(drop=True)

    if not taxa_meta_df.empty:
        metrics_df = metrics_df.merge(taxa_meta_df, on="taxon", how="left")
        contrib_df = contrib_df.merge(taxa_meta_df[["taxon", "order", "family"]], on="taxon", how="left")
        preds_long_df = preds_long_df.merge(taxa_meta_df[["taxon", "order", "family"]], on="taxon", how="left")
    if not food_web_df.empty:
        metrics_df = metrics_df.merge(food_web_df, on=["order", "family"], how="left")
        preds_long_df = preds_long_df.merge(food_web_df, on=["order", "family"], how="left")

    metrics_df["family_unit"] = metrics_df["taxon"]
    roc_df["family_unit"] = roc_df["taxon"]
    contrib_df["family_unit"] = contrib_df["taxon"]
    preds_long_df["family_unit"] = preds_long_df["taxon"]

    indicator_def_rows = []
    for indicator, rule in INDICATOR_DEFINITIONS.items():
        indicator_def_rows.append(
            {
                "indicator_name": indicator,
                "components": " | ".join([str(c) for c in rule["components"]]),                       
                "use_zscore": bool(rule["use_zscore"]),
            }
        )
    indicator_def_df = pd.DataFrame(indicator_def_rows)

    site_mapping_df.to_csv(paths.output_dir / "sdata1_sheet_mapping.csv", index=False, encoding="utf-8-sig")
    indicator_def_df.to_csv(paths.output_dir / "indicator_definition.csv", index=False, encoding="utf-8-sig")
    env_df.to_csv(paths.output_dir / "sdata2_indicator_values.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(paths.output_dir / "model_metrics_auc.csv", index=False, encoding="utf-8-sig")
    roc_df.to_csv(paths.output_dir / "roc_curve_points.csv", index=False, encoding="utf-8-sig")
    contrib_df.to_csv(paths.output_dir / "variable_contributions.csv", index=False, encoding="utf-8-sig")
    preds_long_df.to_csv(paths.output_dir / "layer_probabilities_long.csv", index=False, encoding="utf-8-sig")
    preds_wide_df.to_csv(paths.output_dir / "layer_probabilities_wide.csv", index=False, encoding="utf-8-sig")
    writeMethodDescription(
        paths.output_dir,
        feature_count=len(env_df.columns) - 1,
        model_count=int((metrics_df["model_status"] == "ok").sum()),
        sdata1_sheet_count=len(site_mapping_df),
        sdata3_sheet_count=len(profile_by_sheet),
    )

    print(f"建模完成，结果已输出到: {paths.output_dir}")


if __name__ == "__main__":
    main()


