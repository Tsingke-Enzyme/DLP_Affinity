#!/usr/bin/env python3
"""从单位点突变 KD 数据筛选优于野生型的突变，做位点不冲突的饱和多突变组合。

社区方案参考：
  - 蛋白质工程中的 combinatorial / saturation mutagenesis 库构建；
  - Python 标准库 itertools.combinations 做穷举组合（成熟、无额外部署）。
应用场景：将验证过的增益单突变拼成多突变候选库，供亲和力模型批量预测。
潜在风险：位点一多组合数爆炸（2^n）；同数据若混用 escape_fraction（WT=0）会筛不出增益突变。
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

REQUIRED_PRED_COLS = [
    "antibody_seq",
    "antigen_seq",
    "mutation_id",
    "site",
    "wildtype",
    "mutation",
]


@dataclass(frozen=True)
class SingleMutation:
    """单位点突变记录：用于筛选与组合后写回预测输入。"""

    mutation_id: str
    site: int
    wildtype: str
    mutation: str
    antibody_seq: str
    antigen_seq: str
    kd: float
    seq_index: int


def _resolve_label_col(columns: Sequence[str], preferred: str) -> str:
    """解析亲和力标签列名。

    输入：
        columns: DataFrame 列名
        preferred: 优先列名（如 kd）
    输出：
        实际使用的列名
    """
    aliases = [preferred, "kd", "KD", "affinity", "escape_fraction", "label", "pkd", "pKD"]
    for name in aliases:
        if name in columns:
            return name
    raise ValueError(f"找不到标签列（优先 {preferred}），实际列: {list(columns)}")


def _is_wildtype_row(row: pd.Series) -> bool:
    """判断一行是否为野生型参考。"""
    mid = str(row.get("mutation_id", "")).strip().upper()
    if mid in {"WT", "WILDTYPE", "WILD_TYPE", "REF", "REFERENCE"}:
        return True
    mut = row.get("mutation", "")
    if pd.isna(mut) or str(mut).strip() == "":
        site = row.get("site", None)
        if site is None or (isinstance(site, float) and math.isnan(site)) or str(site).strip() in {"", "0"}:
            return True
    return False


def _diff_indices(wt_seq: str, mut_seq: str) -> List[int]:
    """比较野生型与突变体抗原序列，返回发生替换的 0-based 下标。"""
    if len(wt_seq) != len(mut_seq):
        raise ValueError(
            f"抗原序列长度不一致: wt={len(wt_seq)} mut={len(mut_seq)}"
        )
    return [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]


def _infer_seq_index(
    wt_seq: str,
    mut_seq: str,
    site: int,
    wildtype_aa: str,
    mutation_aa: str,
) -> int:
    """推断突变在抗原序列中的下标。

    输入：
        wt_seq / mut_seq: 野生型与单突变抗原序列
        site / wildtype_aa / mutation_aa: 突变注释
    输出：
        0-based 序列下标
    处理逻辑：
        优先用序列 diff；若无 diff 则按 RBD 常见编号 site - 偏移；最后校验氨基酸。
    """
    diffs = _diff_indices(wt_seq, mut_seq)
    if len(diffs) == 1:
        idx = diffs[0]
    elif len(diffs) == 0:
        # 常见 RBD 片段：编号起点 = site - index，用 wt 序列长度估计 offset
        # 若 site 落在 [offset, offset+L)，取 site-offset
        # 无法唯一确定时，在 wt 中搜索 wildtype_aa 且 mut_seq 同位置为 mutation_aa
        candidates = [
            i
            for i, aa in enumerate(wt_seq)
            if aa == wildtype_aa and mut_seq[i] == mutation_aa
        ]
        if len(candidates) == 1:
            idx = candidates[0]
        else:
            # 回退：假设 site 为连续编号且序列覆盖 site..site+L-1
            # offset = site - 在 wt 中首次出现 wildtype_aa 的猜测不可靠；用 site 相对最小 site
            raise ValueError(
                f"无法定位位点 site={site} {wildtype_aa}->{mutation_aa}（序列无差异且候选不唯一）"
            )
    else:
        # 多处差异时，优先匹配注释氨基酸
        matched = [
            i
            for i in diffs
            if wt_seq[i] == wildtype_aa and mut_seq[i] == mutation_aa
        ]
        if len(matched) != 1:
            raise ValueError(
                f"多位点差异且无法唯一匹配 site={site} {wildtype_aa}->{mutation_aa}: {diffs}"
            )
        idx = matched[0]

    if wt_seq[idx] != wildtype_aa:
        raise ValueError(
            f"位点校验失败 site={site}: wt[{idx}]={wt_seq[idx]} != {wildtype_aa}"
        )
    if mut_seq[idx] != mutation_aa:
        raise ValueError(
            f"位点校验失败 site={site}: mut[{idx}]={mut_seq[idx]} != {mutation_aa}"
        )
    return idx


def load_wildtype(
    single_df: pd.DataFrame,
    wt_path: Optional[str],
    label_col: str,
    wt_kd: Optional[float],
) -> Tuple[str, str, float]:
    """加载野生型抗体序列、抗原序列与 KD。

    输入：
        single_df: 单突变表
        wt_path: 可选野生型 CSV
        label_col: 标签列
        wt_kd: 可选显式 KD
    输出：
        (antibody_seq, antigen_seq, wt_kd)
    """
    if wt_path:
        wt_df = pd.read_csv(wt_path)
        if wt_df.empty:
            raise ValueError(f"野生型文件为空: {wt_path}")
        row = wt_df.iloc[0]
        ab = str(row["antibody_seq"])
        ag = str(row["antigen_seq"])
        if wt_kd is not None:
            return ab, ag, float(wt_kd)
        wt_label_col = _resolve_label_col(wt_df.columns, label_col)
        return ab, ag, float(row[wt_label_col])

    wt_rows = single_df[single_df.apply(_is_wildtype_row, axis=1)]
    if not wt_rows.empty:
        row = wt_rows.iloc[0]
        ab = str(row["antibody_seq"])
        ag = str(row["antigen_seq"])
        if wt_kd is not None:
            return ab, ag, float(wt_kd)
        return ab, ag, float(row[label_col])

    if wt_kd is None:
        raise ValueError(
            "未找到野生型行：请提供 --wt-path，或在输入中包含 mutation_id=WT，或指定 --wt-kd"
        )
    # 仅有 wt_kd：从首条单突变的序列 diff 反推 wt 抗原（要求每行都带完整突变体序列）
    row0 = single_df.iloc[0]
    ab = str(row0["antibody_seq"])
    # 无法可靠反推全长 wt，要求用户提供 wt-path
    raise ValueError("仅指定 --wt-kd 不足，请同时提供 --wt-path 或输入中的 WT 行")


def parse_single_mutations(
    df: pd.DataFrame,
    label_col: str,
    wt_ab: str,
    wt_ag: str,
    wt_kd: float,
    better_direction: str,
) -> List[SingleMutation]:
    """筛选亲和力优于野生型的单位点突变。

    输入：
        df: 单突变数据表
        label_col: KD/标签列
        wt_ab / wt_ag / wt_kd: 野生型参考
        better_direction: lower=标签更小更优（KD）；higher=标签更大更优（pKD）
    输出：
        增益单突变列表
    """
    if better_direction not in {"lower", "higher"}:
        raise ValueError("better_direction 必须是 lower 或 higher")

    mutants: List[SingleMutation] = []
    for _, row in df.iterrows():
        if _is_wildtype_row(row):
            continue
        kd = float(row[label_col])
        if better_direction == "lower":
            if not (kd < wt_kd):
                continue
        else:
            if not (kd > wt_kd):
                continue

        site = int(row["site"])
        wildtype_aa = str(row["wildtype"]).strip()
        mutation_aa = str(row["mutation"]).strip()
        if not wildtype_aa or not mutation_aa or wildtype_aa == mutation_aa:
            continue
        ab = str(row["antibody_seq"])
        ag = str(row["antigen_seq"])
        if ab != wt_ab:
            # 允许空白差异外的不一致时报错，避免把不同抗体混组
            raise ValueError(
                f"antibody_seq 与野生型不一致: mutation_id={row.get('mutation_id')}"
            )
        mid = str(row.get("mutation_id", f"{wildtype_aa}{site}{mutation_aa}"))
        idx = _infer_seq_index(wt_ag, ag, site, wildtype_aa, mutation_aa)
        mutants.append(
            SingleMutation(
                mutation_id=mid,
                site=site,
                wildtype=wildtype_aa,
                mutation=mutation_aa,
                antibody_seq=ab,
                antigen_seq=ag,
                kd=kd,
                seq_index=idx,
            )
        )
    return mutants


def _sites_conflict(combo: Sequence[SingleMutation]) -> bool:
    """同一组合内是否存在位点冲突。"""
    sites = [m.site for m in combo]
    return len(sites) != len(set(sites))


def enumerate_combinations(
    mutants: Sequence[SingleMutation],
    min_order: int,
    max_order: int,
    max_combinations: int,
    seed: int,
) -> List[Tuple[SingleMutation, ...]]:
    """对增益单突变做位点不冲突的饱和（可选抽样）组合。

    输入：
        mutants: 增益单突变
        min_order / max_order: 组合阶数范围（多突变通常从 2 起）
        max_combinations: >0 时对总库随机子采样上限；0 表示穷举
        seed: 随机种子
    输出：
        多突变组合列表（每一项为 SingleMutation 元组）
    """
    n = len(mutants)
    if n == 0:
        return []
    lo = max(2, min_order)
    hi = min(max_order, n) if max_order > 0 else n
    if hi < lo:
        return []

    selected: List[Tuple[SingleMutation, ...]] = []
    for order in range(lo, hi + 1):
        for combo in itertools.combinations(mutants, order):
            if _sites_conflict(combo):
                continue
            selected.append(combo)

    if max_combinations > 0 and len(selected) > max_combinations:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_combinations)
        selected.sort(key=lambda c: (len(c), tuple(m.mutation_id for m in c)))
    return selected


def apply_combo(wt_ag: str, combo: Sequence[SingleMutation]) -> str:
    """将一组位点不冲突的突变施加到野生型抗原序列。"""
    chars = list(wt_ag)
    for m in sorted(combo, key=lambda x: x.seq_index):
        if chars[m.seq_index] != m.wildtype:
            raise ValueError(
                f"施加突变前校验失败 {m.mutation_id}: pos={m.seq_index} "
                f"have={chars[m.seq_index]} expect_wt={m.wildtype}"
            )
        chars[m.seq_index] = m.mutation
    return "".join(chars)


def combinations_to_dataframe(
    combos: Iterable[Sequence[SingleMutation]],
    wt_ab: str,
    wt_ag: str,
) -> pd.DataFrame:
    """将多突变组合整理为 predict.py 可读的 model_input CSV 格式。"""
    rows = []
    for combo in combos:
        ordered = tuple(sorted(combo, key=lambda m: m.site))
        antigen = apply_combo(wt_ag, ordered)
        mutation_id = "+".join(m.mutation_id for m in ordered)
        sites = ";".join(str(m.site) for m in ordered)
        wts = ";".join(m.wildtype for m in ordered)
        muts = ";".join(m.mutation for m in ordered)
        rows.append(
            {
                "antibody_seq": wt_ab,
                "antigen_seq": antigen,
                "mutation_id": mutation_id,
                "site": sites,
                "wildtype": wts,
                "mutation": muts,
            }
        )
    df = pd.DataFrame(rows, columns=REQUIRED_PRED_COLS)
    return df


def build_library(
    input_path: str,
    output_path: str,
    wt_path: Optional[str] = None,
    label_col: str = "kd",
    wt_kd: Optional[float] = None,
    better_direction: str = "lower",
    min_order: int = 2,
    max_order: int = 0,
    max_combinations: int = 0,
    seed: int = 42,
    selected_singles_out: Optional[str] = None,
) -> dict:
    """构建多突变预测输入库。

    输入：
        input_path: 单位点突变验证 KD CSV
        output_path: 预测模块输入 CSV
        wt_path / wt_kd / label_col / better_direction: 野生型与优劣判定
        min_order / max_order / max_combinations / seed: 组合控制
        selected_singles_out: 可选，写出筛选后的增益单突变表
    输出：
        统计摘要 dict
    """
    df = pd.read_csv(input_path)
    for col in ("antibody_seq", "antigen_seq", "site", "wildtype", "mutation"):
        if col not in df.columns:
            raise ValueError(f"输入缺少列 {col}，实际列: {list(df.columns)}")
    resolved_label = _resolve_label_col(df.columns, label_col)
    wt_ab, wt_ag, resolved_wt_kd = load_wildtype(df, wt_path, resolved_label, wt_kd)

    mutants = parse_single_mutations(
        df, resolved_label, wt_ab, wt_ag, resolved_wt_kd, better_direction
    )
    if selected_singles_out:
        pd.DataFrame(
            [
                {
                    "antibody_seq": m.antibody_seq,
                    "antigen_seq": m.antigen_seq,
                    "mutation_id": m.mutation_id,
                    "site": m.site,
                    "wildtype": m.wildtype,
                    "mutation": m.mutation,
                    resolved_label: m.kd,
                    "seq_index": m.seq_index,
                }
                for m in mutants
            ]
        ).to_csv(selected_singles_out, index=False)

    combos = enumerate_combinations(
        mutants, min_order, max_order, max_combinations, seed
    )
    out_df = combinations_to_dataframe(combos, wt_ab, wt_ag)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    summary = {
        "n_input_rows": int(len(df)),
        "wt_kd": resolved_wt_kd,
        "label_col": resolved_label,
        "better_direction": better_direction,
        "n_beneficial_singles": len(mutants),
        "n_combinations": int(len(out_df)),
        "output_path": str(out_path),
    }
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="筛选优于 WT 的单突变并构建饱和多突变预测输入 CSV"
    )
    p.add_argument("--input", required=True, help="单位点突变 KD CSV")
    p.add_argument("--output", required=True, help="多突变 predict 输入 CSV")
    p.add_argument("--wt-path", default="", help="野生型参考 CSV（可选）")
    p.add_argument("--label-col", default="kd", help="亲和力标签列名")
    p.add_argument("--wt-kd", type=float, default=None, help="显式野生型 KD")
    p.add_argument(
        "--better-direction",
        choices=["lower", "higher"],
        default="lower",
        help="lower: KD 越小越好；higher: pKD 越大越好",
    )
    p.add_argument("--min-order", type=int, default=2, help="最小组合阶数")
    p.add_argument(
        "--max-order",
        type=int,
        default=0,
        help="最大组合阶数，0 表示不限制（直到全部增益位点）",
    )
    p.add_argument(
        "--max-combinations",
        type=int,
        default=0,
        help="组合总数上限，0=穷举；超出则按 seed 随机抽样",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--selected-singles-out",
        default="",
        help="可选：写出筛选后的增益单突变 CSV",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口：构建多突变库并打印摘要。"""
    args = parse_args(argv)
    summary = build_library(
        input_path=args.input,
        output_path=args.output,
        wt_path=args.wt_path or None,
        label_col=args.label_col,
        wt_kd=args.wt_kd,
        better_direction=args.better_direction,
        min_order=args.min_order,
        max_order=args.max_order,
        max_combinations=args.max_combinations,
        seed=args.seed,
        selected_singles_out=args.selected_singles_out or None,
    )
    print("=== build_multimutant_library summary ===")
    for k, v in summary.items():
        print(f"{k}={v}")
    if summary["n_beneficial_singles"] == 0:
        print(
            "WARNING: 未筛到优于 WT 的单突变；请检查标签列/better-direction/WT KD",
            file=sys.stderr,
        )
    if summary["n_combinations"] == 0:
        print("ERROR: 多突变组合数为 0，无法继续预测", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
