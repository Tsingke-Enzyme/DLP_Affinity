#!/usr/bin/env python3
"""整理开源抗体–抗原亲和力（K_D）数据到 data/public_data，并汇总训练表。

输入：
  - 已下载的原始 CSV（Zenodo TDC/SAbDab、SKEMPI v2）或仓库内 sdAb 表
  - 现有 demo 训练表 data/demo_data/train_data/{blz_*,mouse}_train_model_input.csv
输出：
  - data/public_data/<db>/README.md + 原始/清洗 CSV + 可训练格式（若有序列）
  - data/demo_data/train_data/all_train_model_input.csv（demo + 可训练开源行）
处理逻辑：
  仅将「antibody_seq + antigen_seq + kd(M)」齐全的行并入 all_train；
  SKEMPI/sdAb 缺抗原序列时只落盘公开库并在 README 说明，不硬拼训练表。
"""

from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "public_data"
DEMO_TRAIN = ROOT / "data" / "demo_data" / "train_data"
RELEASE_SDAB = ROOT / "release_package" / "data" / "sdab_nanobody_data2.csv"

# 官方/镜像下载 URL（优先国内可访问的 Zenodo / BSC）
URL_TDC_SABDAB = (
    "https://zenodo.org/records/13120765/files/"
    "antibody_affinity_protein_sabdab.csv?download=1"
)
URL_SKEMPI_V2 = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"

OUT_COLS = [
    "antibody_seq",
    "antigen_seq",
    "mutation_id",
    "site",
    "wildtype",
    "mutation",
    "kd",
    "dataset_source",
]

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.I)


def _clean_aa(seq: str) -> str:
    """去掉空白与非标准字符，保留标准 20 氨基酸大写串。"""
    s = re.sub(r"\s+", "", str(seq).upper())
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s)
    return s


def _is_valid_seq(seq: str, min_len: int = 5) -> bool:
    """序列是否可作为模型输入（长度与字母表）。"""
    return bool(seq) and len(seq) >= min_len and bool(AA_RE.match(seq))


def _write_readme(path: Path, lines: List[str]) -> None:
    """写入 README；调用方应对含反斜杠的 LaTeX 使用 raw 字符串。"""
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _download(url: str, dest: Path) -> Path:
    """下载 url 到 dest（若已存在且非空则跳过）。

    输入：url 下载地址；dest 本地路径。
    输出：dest 路径。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        print(f"[skip-download] {dest} exists ({dest.stat().st_size} bytes)")
        return dest
    import urllib.request

    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    if not dest.is_file() or dest.stat().st_size < 100:
        raise RuntimeError(f"下载失败或文件过小: {dest}")
    return dest


def prepare_tdc_sabdab(src_csv: Path) -> pd.DataFrame:
    """整理 Zenodo/TDC Protein_SAbDab（Antibody+Antigen+Y=Kd）。

    输入：原始 CSV 路径。
    输出：训练格式 DataFrame（kd 单位 M）。
    """
    out_dir = PUBLIC / "tdc_sabdab"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst_raw = raw_dir / "antibody_affinity_protein_sabdab.csv"
    if Path(src_csv).resolve() != dst_raw.resolve():
        shutil.copy2(src_csv, dst_raw)

    raw = pd.read_csv(dst_raw)
    rows: List[Dict] = []
    skipped = 0
    for i, r in raw.iterrows():
        try:
            ab_parts = ast.literal_eval(str(r["Antibody"]))
            if isinstance(ab_parts, list):
                ab = "".join(_clean_aa(p) for p in ab_parts)
            else:
                ab = _clean_aa(ab_parts)
        except (ValueError, SyntaxError):
            ab = _clean_aa(str(r["Antibody"]))
        ag = _clean_aa(str(r["Antigen"]))
        try:
            kd = float(r["Y"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        if kd <= 0 or not _is_valid_seq(ab, 20) or not _is_valid_seq(ag, 5):
            skipped += 1
            continue
        mid = str(r.get("Antibody_ID", f"sabdab_{i}"))
        rows.append(
            {
                "antibody_seq": ab,
                "antigen_seq": ag,
                "mutation_id": mid,
                "site": "",
                "wildtype": "",
                "mutation": "",
                "kd": kd,
                "dataset_source": "tdc_sabdab",
            }
        )

    df = pd.DataFrame(rows, columns=OUT_COLS)
    train_path = out_dir / "train_model_input.csv"
    df.to_csv(train_path, index=False)

    _write_readme(
        out_dir / "README.md",
        [
            r"# TDC / SAbDab Protein_SAbDab（抗体–抗原 $K_D$）",
            "",
            "## 来源",
            "- **上游数据库**：SAbDab（Structural Antibody Database，Oxford OPIG）",
            "- **ML 整理**：Therapeutics Data Commons（TDC）任务 `AntibodyAff` / `Protein_SAbDab`",
            "- **本仓库下载镜像**：Zenodo DOI [10.5281/zenodo.13120765](https://doi.org/10.5281/zenodo.13120765)",
            "  （`antibody_affinity_protein_sabdab.csv`，由 TDC 导出）",
            "- **文献**：Dunbar et al., *NAR* 2014. DOI: [10.1093/nar/gkt1043](https://doi.org/10.1093/nar/gkt1043)",
            "- **许可**：CC BY 3.0（TDC 标注）",
            "",
            "## 数据说明",
            "| 字段 | 含义 |",
            "|------|------|",
            "| `Antibody` | 抗体序列；原表为 Python list 字符串，VH+VL 拼接 |",
            "| `Antigen` | 抗原氨基酸序列（蛋白/肽） |",
            r"| `Y` | 结合亲和力 **$K_D$**，单位 **M（mol/L）** |",
            "",
            f"- 原始行数：{len(raw)}",
            rf"- 清洗后可训练行数：{len(df)}（跳过 {skipped}：非法序列或非正 $K_D$）",
            rf"- $K_D$ 范围（M）：{df['kd'].min():.3e} ~ {df['kd'].max():.3e}",
            "",
            "## 本目录文件",
            "- `raw/antibody_affinity_protein_sabdab.csv`：原始下载",
            "- `train_model_input.csv`：DLP-Affinity 训练列格式（`kd` 已为 M）",
            "",
            "## 使用注意",
            r"- TDC 文档建议对亲和力取对数；本项目训练默认 `log_transform_kd=True`（$\log_{10} K_D$）。",
            "- 抗体列为重链+轻链拼接，与 blz demo 的单链/可变区长度分布可能不同。",
            "",
        ],
    )
    print(f"[ok] tdc_sabdab: raw={len(raw)} train={len(df)} -> {train_path}")
    return df


def prepare_skempi(src_csv: Path) -> None:
    """整理 SKEMPI 2.0：保留 AB/AG 子集的 Kd 表（无序列，不并入训练）。"""
    out_dir = PUBLIC / "skempi_v2"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst_raw = raw_dir / "skempi_v2.csv"
    if Path(src_csv).resolve() != dst_raw.resolve():
        shutil.copy2(src_csv, dst_raw)

    raw = pd.read_csv(dst_raw, sep=";")
    # Hold_out_type 含 AB/AG 的抗体–抗原条目
    hold = raw["Hold_out_type"].fillna("").astype(str)
    abag = raw[hold.str.contains("AB/AG", na=False)].copy()

    def parse_kd(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    abag["kd_mut_M"] = parse_kd(abag["Affinity_mut (M)"])
    abag["kd_wt_M"] = parse_kd(abag["Affinity_wt (M)"])
    abag_out = abag[
        [
            "#Pdb",
            "Mutation(s)_cleaned",
            "iMutation_Location(s)",
            "Hold_out_type",
            "Protein 1",
            "Protein 2",
            "Affinity_mut (M)",
            "Affinity_wt (M)",
            "kd_mut_M",
            "kd_wt_M",
            "Reference",
            "Temperature",
        ]
    ].copy()
    abag_path = out_dir / "ab_ag_affinity.csv"
    abag_out.to_csv(abag_path, index=False)

    n_mut = int(abag_out["kd_mut_M"].notna().sum())
    n_wt = int(abag_out["kd_wt_M"].notna().sum())

    _write_readme(
        out_dir / "README.md",
        [
            r"# SKEMPI 2.0（含抗体–抗原子集的 $K_D$）",
            "",
            "## 来源",
            "- **官网**：https://life.bsc.es/pid/skempi2/",
            "- **下载**：`skempi_v2.csv`（分号分隔）",
            "- **文献**：Jankauskaitė et al., *Bioinformatics* 2019. DOI: [10.1093/bioinformatics/bty635](https://doi.org/10.1093/bioinformatics/bty635)",
            "",
            "## 数据说明",
            "SKEMPI 收录蛋白–蛋白复合物突变对结合自由能/动力学的影响。",
            "与抗体相关的子集通过 `Hold_out_type` 含 `AB/AG` 筛选。",
            "",
            "| 关键字段 | 含义 |",
            "|----------|------|",
            "| `#Pdb` | PDB ID + 链标注 |",
            r"| `Affinity_mut (M)` / `Affinity_wt (M)` | 突变体 / 野生型 **$K_D$**（单位 M） |",
            "| `Mutation(s)_cleaned` | 突变标注 |",
            "| `Protein 1` / `Protein 2` | 伙伴蛋白名称 |",
            "",
            f"- 全库行数：{len(raw)}",
            f"- AB/AG 子集行数：{len(abag_out)}",
            f"- 可解析 `kd_mut_M`：{n_mut}；`kd_wt_M`：{n_wt}",
            "",
            "## 本目录文件",
            "- `raw/skempi_v2.csv`：官方全表",
            r"- `ab_ag_affinity.csv`：抗体–抗原子集 + 数值化 $K_D$ 列",
            "",
            "## 为何未并入 `all_train_model_input.csv`",
            "SKEMPI **不提供氨基酸序列**，仅有 PDB ID。要进入 DLP-Affinity 训练需从结构抽取抗体/抗原链序列。",
            r"本步骤仅归档 $K_D$ 与元数据；后续可另写 PDB→序列脚本再合并。",
            "",
        ],
    )
    print(f"[ok] skempi_v2: ab_ag={len(abag_out)} kd_mut={n_mut} -> {abag_path}")


def prepare_sdab(src_csv: Path) -> None:
    """整理仓库内 sdAb nanobody 表（Kd nM；抗原多为名称无序列）。"""
    out_dir = PUBLIC / "sdab_nanobody"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst_raw = raw_dir / "sdab_nanobody_data2.csv"
    if Path(src_csv).resolve() != dst_raw.resolve():
        shutil.copy2(src_csv, dst_raw)

    raw = pd.read_csv(dst_raw)
    kd_col = "Affinity,KD(nM)"
    rows = []
    for _, r in raw.iterrows():
        raw_kd = r.get(kd_col)
        if pd.isna(raw_kd):
            continue
        s = str(raw_kd).strip()
        if s.lower() in {"", "none", "nan", "no-binding", "no_binding"}:
            continue
        try:
            kd_nm = float(s)
        except ValueError:
            continue
        if kd_nm <= 0:
            continue
        ab = _clean_aa(str(r.get("Sequence", "")))
        if not _is_valid_seq(ab, 50):
            continue
        rows.append(
            {
                "antibody_name": r.get("Antibody name", ""),
                "antibody_source": r.get("Antibody source", ""),
                "target_antigen_name": r.get("Target antigen", ""),
                "antibody_seq": ab,
                "kd_nM": kd_nm,
                "kd_M": kd_nm * 1e-9,
                "dataset_source": "sdab_nanobody",
            }
        )
    cleaned = pd.DataFrame(rows)
    cleaned_path = out_dir / "cleaned_with_kd.csv"
    cleaned.to_csv(cleaned_path, index=False)

    _write_readme(
        out_dir / "README.md",
        [
            r"# sdAb / Nanobody 亲和力表（$K_D$ nM）",
            "",
            "## 来源",
            "- 文件来自本仓库 `release_package/data/sdab_nanobody_data2.csv`",
            "- 内容为单域抗体（nanobody / sdAb）文献汇总之亲和力与序列",
            "- 原始列名含 `Affinity,KD(nM)`，单位为 **nM**",
            "",
            "## 数据说明",
            "| 字段 | 含义 |",
            "|------|------|",
            "| `Sequence` | nanobody 氨基酸序列 |",
            r"| `Affinity,KD(nM)` | **$K_D$**，单位 nM；清洗后另存 `kd_M = kd_nM × 10⁻⁹` |",
            "| `Target antigen` | 抗原**名称**（非序列） |",
            "",
            f"- 原始行数：{len(raw)}",
            rf"- 有效 $K_D$+序列行数：{len(cleaned)}",
            "",
            "## 本目录文件",
            "- `raw/sdab_nanobody_data2.csv`：原始拷贝",
            r"- `cleaned_with_kd.csv`：过滤后的抗体序列与 $K_D$（nM 与 M）",
            "",
            "## 为何未并入 `all_train_model_input.csv`",
            "缺少 `antigen_seq`。DLP-Affinity 前向需要抗体与抗原两条序列。",
            "补全抗原序列（如 UniProt/PDB）后方可合并。",
            "",
        ],
    )
    print(f"[ok] sdab_nanobody: cleaned={len(cleaned)} -> {cleaned_path}")


def load_demo_frames() -> List[pd.DataFrame]:
    """加载现有 demo 分类型训练表。"""
    frames: List[pd.DataFrame] = []
    for name in ("blz_3n", "blz_4n", "mouse"):
        p = DEMO_TRAIN / f"{name}_train_model_input.csv"
        if not p.is_file():
            print(f"[warn] missing demo table: {p}")
            continue
        df = pd.read_csv(p)
        frames.append(df[OUT_COLS] if set(OUT_COLS).issubset(df.columns) else df)
        print(f"[ok] demo {name}: {len(df)}")
    return frames


def merge_all_train(public_trainable: List[pd.DataFrame]) -> pd.DataFrame:
    """合并 demo + 可训练开源表，写出 all_train_model_input.csv。"""
    frames = load_demo_frames() + public_trainable
    if not frames:
        raise RuntimeError("无可用训练表可合并")
    all_df = pd.concat(frames, ignore_index=True)
    # 去重：同序列+同 kd+同来源
    before = len(all_df)
    all_df = all_df.drop_duplicates(
        subset=["antibody_seq", "antigen_seq", "kd", "dataset_source"], keep="first"
    )
    out = DEMO_TRAIN / "all_train_model_input.csv"
    DEMO_TRAIN.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out, index=False)

    counts = all_df["dataset_source"].value_counts().to_dict()
    summary_lines = [f"- `{k}`: n={v}" for k, v in sorted(counts.items())]
    readme = DEMO_TRAIN / "README.md"
    _write_readme(
        readme,
        [
            r"# 训练格式汇总（demo + 开源 $K_D$）",
            "",
            "由 `script/prepare_demo_train_data.py` 生成 demo 部分，",
            "再由 `script/prepare_public_affinity_data.py` 并入开源可训练行。",
            "",
            "## 列说明",
            "",
            "| 列 | 含义 |",
            "|----|------|",
            "| `antibody_seq` | 抗体氨基酸序列 |",
            "| `antigen_seq` | 抗原氨基酸序列 |",
            "| `mutation_id` | 样本 ID |",
            "| `site` / `wildtype` / `mutation` | 位点注释；WT 为 `mutation_id=WT,site=0` |",
            r"| `kd` | **$K_D$**，单位 **M**（未取对数） |",
            "| `dataset_source` | 来源：`blz_3n` / `blz_4n` / `mouse` / `tdc_sabdab` 等 |",
            "",
            "## 文件",
            "",
            f"- **全量汇总**：`all_train_model_input.csv`（n={len(all_df)}，去重前 {before}）",
            *summary_lines,
            "",
            "## 开源数据说明",
            "",
            "详见 `data/public_data/*/README.md`。",
            r"仅 **抗体序列 + 抗原序列 + $K_D$(M)** 齐全的开源行并入本表；",
            "SKEMPI（无序列）、sdAb（无抗原序列）仅归档在 `public_data/`。",
            "",
            "> 注意：`data/` 目录在 `.gitignore` 中；投递集群前请同步到 NAS。",
            "",
        ],
    )
    print(f"[ok] all_train: {len(all_df)} -> {out}")
    print("[ok] sources:", counts)
    return all_df


def write_public_index() -> None:
    """写 public_data 总 README。"""
    _write_readme(
        PUBLIC / "README.md",
        [
            r"# 开源抗体–抗原亲和力（$K_D$）公开数据",
            "",
            "由 `script/prepare_public_affinity_data.py` 下载/整理。",
            "",
            "## 收录库",
            "",
            "| 目录 | 标签类型 | 有 ab+ag 序列 | 并入 all_train |",
            "|------|----------|---------------|----------------|",
            r"| `tdc_sabdab/` | $K_D$ (M) | 是 | 是 |",
            r"| `skempi_v2/` | $K_D$ (M) | 否（仅 PDB） | 否 |",
            r"| `sdab_nanobody/` | $K_D$ (nM→M) | 仅抗体 | 否 |",
            "",
            "## 未收录说明",
            "",
            r"- **AB-Bind**：仓库 `release_package/data/AB-Bind/` 已有，标签主要为 **$\Delta\Delta G$**，非绝对 $K_D$。",
            r"- **7KMG DMS**：`escape_fraction`，非物理 $K_D$。",
            "- **AbDesign DB**：需 Google Drive 人工下载，本脚本未自动拉取。",
            "",
            "## 再生成",
            "",
            "```bash",
            "python3 script/prepare_public_affinity_data.py",
            "```",
            "",
        ],
    )


def main() -> None:
    """入口：下载（如需）→整理公开库→汇总训练表。"""
    PUBLIC.mkdir(parents=True, exist_ok=True)

    sab_src = PUBLIC / "tdc_sabdab" / "raw" / "antibody_affinity_protein_sabdab.csv"
    tmp_sab = Path("/tmp/kd_dl/antibody_affinity_protein_sabdab.csv")
    if not sab_src.is_file():
        if tmp_sab.is_file():
            sab_src.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_sab, sab_src)
        else:
            _download(URL_TDC_SABDAB, sab_src)

    sk_src = PUBLIC / "skempi_v2" / "raw" / "skempi_v2.csv"
    tmp_sk = Path("/tmp/kd_dl/skempi_v2.csv")
    if not sk_src.is_file():
        if tmp_sk.is_file():
            sk_src.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_sk, sk_src)
        else:
            try:
                _download(URL_SKEMPI_V2, sk_src)
            except Exception as exc:
                print(f"[warn] skempi download failed: {exc}")

    sab_df = prepare_tdc_sabdab(sab_src)
    if sk_src.is_file():
        prepare_skempi(sk_src)
    else:
        print("[warn] skip skempi: source not found")

    if RELEASE_SDAB.is_file():
        prepare_sdab(RELEASE_SDAB)
    else:
        print("[warn] skip sdab: release_package file missing")

    write_public_index()
    merge_all_train([sab_df])


if __name__ == "__main__":
    main()
