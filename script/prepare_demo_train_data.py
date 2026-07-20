#!/usr/bin/env python3
"""将 demo_data 各类型 single_mutations.csv + PDB 转为训练投递格式。

输入：
  demo_data/{blz_3n,blz_4n,mouse}/single_mutations.csv 与 *.pdb
输出：
  1) 各类型目录下 single_samples/：每行一个训练格式 CSV
  2) demo_data/train_data/：全量汇总 CSV 及按类型拆分的汇总 CSV
处理逻辑：
  PDB 链 B → antibody_seq（与 mut_seq 一致），链 A → antigen_seq；
  标签列 kd（原始摩尔浓度 K_D）；mutation「site:AA」解析 site/wildtype/mutation。
  野生型行对齐 7KMG / `_is_wildtype_row` 可识别格式：
  mutation_id=WT, site=0, wildtype/mutation 为空（源表 id=parent、mutation=WT 仅作输入）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio.PDB import PDBParser, PPBuilder

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo_data"
TRAIN_OUT = DEMO / "train_data"
DATA_TYPES = ("blz_3n", "blz_4n", "mouse")

# 与 Argo 训练 / AffinityDataset 兼容的列（kd 为原始尺度）
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


def pdb_chain_seq(pdb_path: Path, chain_id: str) -> str:
    """从 PDB 提取指定链氨基酸序列。

    输入：pdb_path 结构文件路径；chain_id 链 ID。
    输出：单字母氨基酸串。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    ppb = PPBuilder()
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                return "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(chain))
    raise ValueError(f"chain {chain_id} not found in {pdb_path}")


def find_pdb(type_dir: Path) -> Path:
    """在类型目录中定位唯一 PDB。"""
    pdbs = sorted(type_dir.glob("*.pdb"))
    if not pdbs:
        raise FileNotFoundError(f"no PDB under {type_dir}")
    return pdbs[0]


def parse_mutation(
    mutation_field: str, parent_seq: str
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """解析 mutation 字段。

    输入：如 ``WT`` 或 ``93:P``；parent_seq 野生型序列。
    输出：(site, wildtype, mutation_aa)；WT 时三者为 None。
    """
    if mutation_field == "WT" or not mutation_field or pd.isna(mutation_field):
        return None, None, None
    m = re.fullmatch(r"(\d+):([A-Z])", str(mutation_field).strip())
    if not m:
        raise ValueError(f"unsupported mutation format: {mutation_field!r}")
    site = int(m.group(1))
    mut_aa = m.group(2)
    if site < 1 or site > len(parent_seq):
        raise ValueError(f"site {site} out of range for parent len={len(parent_seq)}")
    wt_aa = parent_seq[site - 1]
    return site, wt_aa, mut_aa


def safe_filename(sample_id: str) -> str:
    """将样本 id 转为安全文件名。"""
    name = re.sub(r"[^\w.\-]+", "_", str(sample_id)).strip("._")
    return name or "sample"


def convert_type(type_name: str) -> pd.DataFrame:
    """转换单一数据类型为训练格式 DataFrame。"""
    type_dir = DEMO / type_name
    csv_path = type_dir / "single_mutations.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    raw = pd.read_csv(csv_path)
    pdb_path = find_pdb(type_dir)
    antigen_seq = pdb_chain_seq(pdb_path, "A")
    antibody_wt = pdb_chain_seq(pdb_path, "B")

    parent_rows = raw[raw["mutation"].astype(str) == "WT"]
    if parent_rows.empty:
        raise ValueError(f"{type_name}: missing WT/parent row")
    parent_seq = str(parent_rows.iloc[0]["mut_seq"])
    if parent_seq != antibody_wt:
        raise ValueError(
            f"{type_name}: PDB chain B != WT mut_seq "
            f"(pdb_len={len(antibody_wt)} csv_len={len(parent_seq)})"
        )

    rows: List[Dict] = []
    for _, r in raw.iterrows():
        mut_seq = str(r["mut_seq"])
        site, wt_aa, mut_aa = parse_mutation(str(r["mutation"]), parent_seq)
        if site is not None and mut_seq[site - 1] != mut_aa:
            raise ValueError(
                f"{type_name}/{r['id']}: mut_seq[{site}]={mut_seq[site-1]} != {mut_aa}"
            )
        is_wt = mut_aa is None
        # WT：mutation_id=WT / site=0 / mutation 空，供 build_multimutant_library 识别
        rows.append(
            {
                "antibody_seq": mut_seq,
                "antigen_seq": antigen_seq,
                "mutation_id": "WT" if is_wt else str(r["id"]),
                "site": 0 if is_wt else site,
                "wildtype": "" if is_wt else wt_aa,
                "mutation": "" if is_wt else mut_aa,
                "kd": float(r["kd"]),
                "dataset_source": type_name,
            }
        )

    df = pd.DataFrame(rows, columns=OUT_COLS)

    # 单样本目录
    samples_dir = type_dir / "single_samples"
    if samples_dir.exists():
        for old in samples_dir.glob("*.csv"):
            old.unlink()
    samples_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        out = samples_dir / f"{safe_filename(row['mutation_id'])}.csv"
        pd.DataFrame([row])[OUT_COLS].to_csv(out, index=False)

    # 类型内全表（训练格式）
    type_train = type_dir / "train_model_input.csv"
    df.to_csv(type_train, index=False)
    return df


def main() -> None:
    """汇总全部 demo 类型数据并写出 train_data。"""
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    summary: List[str] = []

    for name in DATA_TYPES:
        df = convert_type(name)
        frames.append(df)
        per_path = TRAIN_OUT / f"{name}_train_model_input.csv"
        df.to_csv(per_path, index=False)
        summary.append(f"- `{name}`: n={len(df)} → `{per_path.name}` 与 `{name}/single_samples/`")
        print(f"[ok] {name}: {len(df)} rows, single_samples={len(list((DEMO/name/'single_samples').glob('*.csv')))}")

    all_df = pd.concat(frames, ignore_index=True)
    all_path = TRAIN_OUT / "all_train_model_input.csv"
    all_df.to_csv(all_path, index=False)

    readme = TRAIN_OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# demo_data 训练格式汇总",
                "",
                "由 `script/prepare_demo_train_data.py` 从各类型 `single_mutations.csv` + PDB 生成。",
                "",
                "## 列说明（Argo 训练可直接使用）",
                "",
                "| 列 | 含义 |",
                "|----|------|",
                "| `antibody_seq` | 抗体氨基酸序列（突变体或 WT） |",
                "| `antigen_seq` | 抗原氨基酸序列（PDB 链 A） |",
                "| `mutation_id` | 样本 ID |",
                "| `site` / `wildtype` / `mutation` | 位点与突变；WT 行对齐 7KMG：`mutation_id=WT,site=0`，wildtype/mutation 为空 |",
                r"| `kd` | 原始 $K_D$（单位 M，未取对数） |",
                "| `dataset_source` | 来源类型目录名 |",
                "",
                "## 文件",
                "",
                f"- **全量汇总**：`{all_path.name}`（n={len(all_df)}）",
                *summary,
                "",
                "各类型目录另有：",
                "- `train_model_input.csv`：该类型全表",
                "- `single_samples/*.csv`：每样本一行的训练格式文件",
                "",
                "## 投递示例",
                "",
                "```bash",
                f"TRAIN_PATH=$(pwd)/demo_data/train_data/all_train_model_input.csv \\",
                f"VAL_PATH=$(pwd)/demo_data/train_data/all_train_model_input.csv \\",
                "./argo/dlp-affinity-train.submit.sh",
                "```",
                "",
                "> 注意：路径须在集群 NAS 挂载前缀内；投递前请同步到 NAS。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[ok] all: {len(all_df)} rows → {all_path}")
    print(f"[ok] README → {readme}")


if __name__ == "__main__":
    main()
