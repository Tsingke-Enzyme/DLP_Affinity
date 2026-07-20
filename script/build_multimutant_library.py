#!/usr/bin/env python3
"""从单位点突变 KD 数据筛选优于野生型的突变，做位点不冲突的饱和多突变组合。

社区方案参考：
  - 蛋白质工程中的 combinatorial / saturation mutagenesis 库构建；
  - Python 标准库 itertools.combinations 做穷举组合（成熟、无额外部署）。
应用场景：将验证过的增益单突变拼成多突变候选库，供亲和力模型批量预测。
支持抗体链或抗原链上的单位点突变（--mutation-target auto|antibody|antigen）。
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
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import pandas as pd

REQUIRED_PRED_COLS = [
    "antibody_seq",
    "antigen_seq",
    "mutation_id",
    "site",
    "wildtype",
    "mutation",
]

SEQ_AB_ALIASES = ("antibody_seq", "seq_ab", "ab_seq")
SEQ_AG_ALIASES = ("antigen_seq", "seq_ag", "ag_seq")
LABEL_ALIASES = ("kd", "KD", "affinity", "escape_fraction", "label", "pkd", "pKD")

# 突变施加链：antibody=抗体工程库；antigen=抗原/DMS（如 7KMG）
MutationTarget = Literal["antibody", "antigen"]


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
    target: MutationTarget


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


def _diff_indices(wt_seq: str, mut_seq: str, role: str = "seq") -> List[int]:
    """比较野生型与突变体序列，返回发生替换的 0-based 下标。

    输入：
        wt_seq / mut_seq: 野生型与突变体氨基酸串
        role: 报错时的链角色名（antibody/antigen/seq）
    输出：
        差异位置列表（0-based）
    """
    if len(wt_seq) != len(mut_seq):
        raise ValueError(
            f"{role} 序列长度不一致: wt={len(wt_seq)} mut={len(mut_seq)}"
        )
    return [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]


def _classify_row_target(
    ab: str, ag: str, wt_ab: str, wt_ag: str
) -> Optional[MutationTarget]:
    """判断一行相对 WT 属于抗体单位点、抗原单位点，或无法归类。

    输入：行序列与野生型序列。
    输出：'antibody' / 'antigen' / None（长度不一致、多位点或双链均变）。
    """
    try:
        ab_diffs = _diff_indices(wt_ab, ab, role="antibody")
        ag_diffs = _diff_indices(wt_ag, ag, role="antigen")
    except ValueError:
        return None
    if len(ab_diffs) == 1 and len(ag_diffs) == 0:
        return "antibody"
    if len(ag_diffs) == 1 and len(ab_diffs) == 0:
        return "antigen"
    return None


def detect_mutation_target(
    df: pd.DataFrame,
    wt_ab: str,
    wt_ag: str,
    requested: str = "auto",
) -> MutationTarget:
    """解析突变目标链。

    输入：
        df: 单点表；wt_ab / wt_ag: 野生型；requested: auto|antibody|antigen
    输出：
        MutationTarget
    处理逻辑：
        auto 时统计「固定链=WT 且突变链恰差 1 位」的行数；仅一侧有票则选定，
        两侧皆有或皆无则报错（禁止静默猜测）。
    """
    if requested in {"antibody", "antigen"}:
        print(f"[mutation-target] {requested} (explicit)")
        return requested  # type: ignore[return-value]
    if requested != "auto":
        raise ValueError(
            f"mutation-target 必须是 auto|antibody|antigen，收到 {requested!r}"
        )

    ab_col, ag_col = _resolve_seq_cols(df)
    n_ab = 0
    n_ag = 0
    for _, row in df.iterrows():
        if _is_wildtype_row(row):
            continue
        kind = _classify_row_target(
            str(row[ab_col]), str(row[ag_col]), wt_ab, wt_ag
        )
        if kind == "antibody":
            n_ab += 1
        elif kind == "antigen":
            n_ag += 1

    if n_ab > 0 and n_ag == 0:
        print(f"[mutation-target] antibody (n={n_ab})")
        return "antibody"
    if n_ag > 0 and n_ab == 0:
        print(f"[mutation-target] antigen (n={n_ag})")
        return "antigen"
    raise ValueError(
        "无法自动判定 mutation-target："
        f"antibody 单位点行={n_ab}, antigen 单位点行={n_ag}；"
        "请用 --mutation-target antibody|antigen 显式指定"
    )


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


def consensus_sequence(sequences: Sequence[str], role: str = "seq") -> str:
    """由多条等长序列按位点多数表决得到一致性序列。

    输入：
        sequences: 氨基酸序列列表
        role: 日志用途标识
    输出：
        一致性序列字符串
    """
    from collections import Counter

    seqs = [str(s) for s in sequences if str(s).strip()]
    if not seqs:
        raise ValueError(f"[{role}] 无可用序列计算一致性")
    length = len(seqs[0])
    if any(len(s) != length for s in seqs):
        raise ValueError(f"[{role}] 序列长度不一致，无法计算一致性序列")
    chars: List[str] = []
    for i in range(length):
        aa, _ = Counter(s[i] for s in seqs).most_common(1)[0]
        chars.append(aa)
    cons = "".join(chars)
    print(f"[consensus] {role}: n={len(seqs)} len={length}")
    return cons


def _resolve_seq_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """解析抗体/抗原序列列名。"""
    return (
        _find_col(df.columns, SEQ_AB_ALIASES, "抗体序列"),
        _find_col(df.columns, SEQ_AG_ALIASES, "抗原序列"),
    )


def find_wildtype_row(df: pd.DataFrame) -> Optional[pd.Series]:
    """在表中查找野生型行；找不到返回 None。"""
    wt_rows = df[df.apply(_is_wildtype_row, axis=1)]
    if wt_rows.empty:
        return None
    return wt_rows.iloc[0]


def resolve_wildtype_reference(
    single_df: pd.DataFrame,
    wt_path: Optional[str],
    label_col: str,
    wt_kd: Optional[float],
) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    """解析野生型参考；若不存在则返回 (None, None, None, 'none')。

    输入：
        single_df / wt_path / label_col / wt_kd
        （仅基于组合输入 --input 与可选 --wt-path，不用扩展训练集）
    输出：
        (antibody_seq, antigen_seq, wt_kd, source)
        source ∈ {wt-path, single, none}
    优先级：wt-path > single 中 WT 行。
    """
    ab_col, ag_col = _resolve_seq_cols(single_df)

    if wt_path:
        wt_df = pd.read_csv(wt_path)
        if wt_df.empty:
            raise ValueError(f"野生型文件为空: {wt_path}")
        row = wt_df.iloc[0]
        wt_ab_col, wt_ag_col = _resolve_seq_cols(wt_df)
        ab = str(row[wt_ab_col])
        ag = str(row[wt_ag_col])
        if wt_kd is not None:
            return ab, ag, float(wt_kd), "wt-path"
        try:
            wt_label_col = _resolve_label_col(wt_df.columns, label_col)
            return ab, ag, float(row[wt_label_col]), "wt-path"
        except ValueError:
            # WT 文件可无标签
            return ab, ag, wt_kd, "wt-path"

    row = find_wildtype_row(single_df)
    if row is not None:
        ab = str(row[ab_col])
        ag = str(row[ag_col])
        if wt_kd is not None:
            return ab, ag, float(wt_kd), "single"
        return ab, ag, float(row[label_col]), "single"

    return None, None, wt_kd, "none"


def infer_wt_from_consensus(df: pd.DataFrame) -> Tuple[str, str]:
    """无野生型行时，用单点菌库序列的一致性序列推断 WT。

    输入：
        df: 单点突变表（不含或忽略 WT 行）
    输出：
        (antibody_consensus, antigen_consensus)
    """
    ab_col, ag_col = _resolve_seq_cols(df)
    mut_df = df[~df.apply(_is_wildtype_row, axis=1)]
    if mut_df.empty:
        raise ValueError("单点数据为空，无法推断一致性野生型序列")
    wt_ab = consensus_sequence(mut_df[ab_col].tolist(), role="antibody")
    wt_ag = consensus_sequence(mut_df[ag_col].tolist(), role="antigen")
    return wt_ab, wt_ag


def _row_to_single_mutation(
    row: pd.Series,
    label_col: str,
    wt_ab: str,
    wt_ag: str,
    ab_col: str,
    ag_col: str,
    *,
    target: MutationTarget,
    strict: bool,
) -> Optional[SingleMutation]:
    """将一行单点数据转为 SingleMutation；无法定位突变则返回 None。

    输入：
        row / label_col / wt_* / 序列列名
        target: 突变施加链（antibody|antigen）
        strict: True 时固定链与 WT 不一致则抛错；False 则跳过该行
    输出：
        SingleMutation 或 None
    处理逻辑：
        固定链必须等于 WT；突变链相对 WT 恰差 1 位；注释列做交叉校验告警。
    """
    if _is_wildtype_row(row):
        return None
    kd = float(row[label_col])
    ab = str(row[ab_col])
    ag = str(row[ag_col])

    if target == "antigen":
        fixed_ok = ab == wt_ab
        fixed_role = "antibody_seq"
        try:
            diffs = _diff_indices(wt_ag, ag, role="antigen")
        except ValueError:
            if strict:
                raise
            return None
        wt_mut_seq, mut_seq = wt_ag, ag
    else:
        fixed_ok = ag == wt_ag
        fixed_role = "antigen_seq"
        try:
            diffs = _diff_indices(wt_ab, ab, role="antibody")
        except ValueError:
            if strict:
                raise
            return None
        wt_mut_seq, mut_seq = wt_ab, ab

    if not fixed_ok:
        if strict:
            raise ValueError(
                f"{fixed_role} 与野生型不一致: mutation_id={row.get('mutation_id')}"
            )
        return None

    if len(diffs) != 1:
        return None
    idx = diffs[0]
    wildtype_aa = wt_mut_seq[idx]
    mutation_aa = mut_seq[idx]
    if wildtype_aa == mutation_aa:
        return None

    site_val = row.get("site", None)
    if site_val is not None and not (isinstance(site_val, float) and math.isnan(site_val)):
        try:
            site = int(site_val)
        except (TypeError, ValueError):
            site = idx + 1
    else:
        site = idx + 1

    ann_wt = str(row.get("wildtype", "") or "").strip()
    ann_mut = str(row.get("mutation", "") or "").strip()
    if ann_wt and ann_wt != wildtype_aa:
        print(
            f"WARNING: site={site} 注释 wildtype={ann_wt} 与序列 {wildtype_aa} 不一致，采用序列"
        )
    if ann_mut and ann_mut != mutation_aa:
        print(
            f"WARNING: site={site} 注释 mutation={ann_mut} 与序列 {mutation_aa} 不一致，采用序列"
        )

    mid = str(row.get("mutation_id", "") or "").strip()
    if not mid or mid.upper() in {"NAN", "NONE"}:
        mid = f"{wildtype_aa}{site}{mutation_aa}"

    return SingleMutation(
        mutation_id=mid,
        site=site,
        wildtype=wildtype_aa,
        mutation=mutation_aa,
        antibody_seq=ab,
        antigen_seq=ag,
        kd=kd,
        seq_index=idx,
        target=target,
    )


def parse_single_mutations(
    df: pd.DataFrame,
    label_col: str,
    wt_ab: str,
    wt_ag: str,
    wt_kd: float,
    better_direction: str,
    target: MutationTarget,
) -> List[SingleMutation]:
    """筛选亲和力优于野生型的单位点突变。"""
    if better_direction not in {"lower", "higher"}:
        raise ValueError("better_direction 必须是 lower 或 higher")

    ab_col, ag_col = _resolve_seq_cols(df)
    mutants: List[SingleMutation] = []
    for _, row in df.iterrows():
        kd = float(row[label_col]) if not _is_wildtype_row(row) else None
        if kd is None:
            continue
        if better_direction == "lower":
            if not (kd < wt_kd):
                continue
        else:
            if not (kd > wt_kd):
                continue
        m = _row_to_single_mutation(
            row,
            label_col,
            wt_ab,
            wt_ag,
            ab_col,
            ag_col,
            target=target,
            strict=True,
        )
        if m is not None:
            mutants.append(m)
    return mutants


def select_top_affinity_singles(
    df: pd.DataFrame,
    label_col: str,
    wt_ab: str,
    wt_ag: str,
    better_direction: str,
    top_n: int,
    target: MutationTarget,
) -> List[SingleMutation]:
    """无野生型 KD 时：按亲和力取 Top-N（位点去重）作为组合备选。

    输入：
        df: 单点突变表
        label_col / wt_ab / wt_ag / better_direction / top_n / target
    输出：
        至多 top_n 个位点互异的最优单突变
    处理逻辑：
        先将全部可解析单位点按亲和力排序，再按位点贪心保留最优一条，截取 top_n。
    """
    if top_n < 1:
        raise ValueError(f"top_n 必须 >= 1，收到 {top_n}")
    if better_direction not in {"lower", "higher"}:
        raise ValueError("better_direction 必须是 lower 或 higher")

    ab_col, ag_col = _resolve_seq_cols(df)
    parsed: List[SingleMutation] = []
    for _, row in df.iterrows():
        m = _row_to_single_mutation(
            row,
            label_col,
            wt_ab,
            wt_ag,
            ab_col,
            ag_col,
            target=target,
            strict=False,
        )
        if m is not None:
            parsed.append(m)

    if not parsed:
        raise ValueError(
            f"无法从单点数据解析出相对 WT 在 {target} 链上恰差 1 位的突变；"
            "请检查序列是否为真正的单位点突变库，或调整 --mutation-target"
        )

    reverse = better_direction == "higher"
    parsed.sort(key=lambda m: m.kd, reverse=reverse)

    selected: List[SingleMutation] = []
    seen_sites = set()
    for m in parsed:
        if m.site in seen_sites:
            continue
        seen_sites.add(m.site)
        selected.append(m)
        if len(selected) >= top_n:
            break

    print(
        f"[top-n] target={target} direction={better_direction} "
        f"candidates_parsed={len(parsed)} selected={len(selected)} top_n={top_n} "
        f"kd_range=[{selected[-1].kd}, {selected[0].kd}]"
    )
    return selected


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


def apply_combo(wt_seq: str, combo: Sequence[SingleMutation]) -> str:
    """将一组位点不冲突的突变施加到野生型目标链序列。

    输入：
        wt_seq: 野生型目标链（抗体或抗原）
        combo: 位点不冲突的 SingleMutation 序列
    输出：
        施加突变后的序列
    """
    chars = list(wt_seq)
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
    target: MutationTarget,
) -> pd.DataFrame:
    """将多突变组合整理为 predict.py 可读的 model_input CSV 格式。

    输入：
        combos / wt_ab / wt_ag / target
    输出：
        含 antibody_seq、antigen_seq 等列的 DataFrame；突变施加到 target 链。
    """
    rows = []
    for combo in combos:
        ordered = tuple(sorted(combo, key=lambda m: m.site))
        if target == "antigen":
            antibody = wt_ab
            antigen = apply_combo(wt_ag, ordered)
        else:
            antibody = apply_combo(wt_ab, ordered)
            antigen = wt_ag
        mutation_id = "+".join(m.mutation_id for m in ordered)
        sites = ";".join(str(m.site) for m in ordered)
        wts = ";".join(m.wildtype for m in ordered)
        muts = ";".join(m.mutation for m in ordered)
        rows.append(
            {
                "antibody_seq": antibody,
                "antigen_seq": antigen,
                "mutation_id": mutation_id,
                "site": sites,
                "wildtype": wts,
                "mutation": muts,
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_PRED_COLS)


def _find_col(columns: Sequence[str], aliases: Sequence[str], role: str) -> str:
    """在列名中按别名列表查找一列。"""
    for name in aliases:
        if name in columns:
            return name
    raise ValueError(f"缺少{role}列（备选 {list(aliases)}），实际列: {list(columns)}")


def validate_affinity_csv(
    path: str,
    *,
    require_label: bool,
    label_col: str = "kd",
    role: str = "data",
) -> dict:
    """校验训练/验证/单突变 CSV 是否可供后续 train/predict 使用。

    输入：
        path: CSV 路径
        require_label: 是否要求亲和力标签列
        label_col: 优先标签列
        role: 日志中的角色名（train/val/single）
    输出：
        校验摘要 dict
    处理逻辑：
        检查文件存在、非空、序列列齐全；标签可解析；序列非空且长度一致；数值标签可转 float。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"[{role}] 文件不存在: {path}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"[{role}] CSV 为空: {path}")

    ab_col = _find_col(df.columns, SEQ_AB_ALIASES, f"{role} 抗体序列")
    ag_col = _find_col(df.columns, SEQ_AG_ALIASES, f"{role} 抗原序列")
    resolved_label = None
    if require_label:
        preferred = [label_col] + [a for a in LABEL_ALIASES if a != label_col]
        resolved_label = _find_col(df.columns, preferred, f"{role} 亲和力标签")

    null_ab = int(df[ab_col].isna().sum())
    null_ag = int(df[ag_col].isna().sum())
    if null_ab or null_ag:
        raise ValueError(f"[{role}] 存在空序列: antibody_na={null_ab} antigen_na={null_ag}")

    ab_lens = df[ab_col].astype(str).map(len)
    ag_lens = df[ag_col].astype(str).map(len)
    if (ab_lens < 1).any() or (ag_lens < 1).any():
        raise ValueError(f"[{role}] 存在长度为 0 的序列")

    if resolved_label is not None:
        try:
            labels = pd.to_numeric(df[resolved_label], errors="raise")
        except Exception as exc:
            raise ValueError(f"[{role}] 标签列 {resolved_label} 无法解析为数值: {exc}") from exc
        if labels.isna().any():
            raise ValueError(f"[{role}] 标签列 {resolved_label} 含 NaN")

    summary = {
        "role": role,
        "path": str(p),
        "n_rows": int(len(df)),
        "ab_col": ab_col,
        "ag_col": ag_col,
        "label_col": resolved_label,
        "ab_len_min": int(ab_lens.min()),
        "ab_len_max": int(ab_lens.max()),
        "ag_len_min": int(ag_lens.min()),
        "ag_len_max": int(ag_lens.max()),
    }
    print(f"[validate] {role}: rows={summary['n_rows']} "
          f"ab_len=[{summary['ab_len_min']},{summary['ab_len_max']}] "
          f"ag_len=[{summary['ag_len_min']},{summary['ag_len_max']}] "
          f"label={resolved_label}")
    return summary


def validate_pipeline_inputs(
    *,
    single_mutant_path: str,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    wt_path: Optional[str] = None,
    label_col: str = "kd",
) -> List[dict]:
    """流水线前置校验：仅校验组合输入 --input（及可选 --wt-path）。

    输入：
        single_mutant_path / wt_path / label_col
        train_path / val_path：可传入但**不校验**（扩展训练集可能异源、长度不一）
    输出：
        校验摘要列表；失败抛异常
    """
    reports: List[dict] = []
    reports.append(
        validate_affinity_csv(
            single_mutant_path,
            require_label=True,
            label_col=label_col,
            role="single-mutant",
        )
    )
    # site/wildtype/mutation 非强制：无 WT 模式下可由一致性序列与序列 diff 推断

    if wt_path:
        reports.append(
            validate_affinity_csv(
                wt_path,
                require_label=False,
                label_col=label_col,
                role="wildtype",
            )
        )

    if train_path:
        print(
            f"[validate] skip train-path (combo 仅用 --input): {train_path}"
        )
    if val_path:
        print(f"[validate] skip val-path (combo 仅用 --input): {val_path}")
    return reports


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
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    top_n_singles: int = 30,
    consensus_wt_out: Optional[str] = None,
    mutation_target: str = "auto",
) -> dict:
    """构建多突变预测输入库（先校验数据，再组合）。

    输入：
        input_path: 单位点突变验证 KD CSV（组合候选与 WT/突变链判定的唯一数据源）
        output_path: 预测模块输入 CSV
        wt_path / wt_kd / label_col / better_direction: 野生型与优劣判定
        top_n_singles: 无 WT 时按亲和力选取的位点去重 Top-N
        min_order / max_order / max_combinations / seed: 组合控制
        selected_singles_out / consensus_wt_out: 可选落盘
        train_path / val_path: 仅透传记录，不参与组合校验与 WT 解析
        mutation_target: auto|antibody|antigen；auto 按序列 diff 判定突变链
    输出：
        统计摘要 dict
    """
    print("=== validate pipeline inputs ===")
    validate_pipeline_inputs(
        single_mutant_path=input_path,
        train_path=train_path,
        val_path=val_path,
        wt_path=wt_path,
        label_col=label_col,
    )

    df = pd.read_csv(input_path)
    ab_col, ag_col = _resolve_seq_cols(df)
    resolved_label = _resolve_label_col(df.columns, label_col)

    wt_ab, wt_ag, resolved_wt_kd, wt_source = resolve_wildtype_reference(
        df, wt_path, resolved_label, wt_kd
    )

    selection_mode: str
    if wt_ab and wt_ag and resolved_wt_kd is not None:
        selection_mode = "better-than-wt"
        target = detect_mutation_target(df, wt_ab, wt_ag, mutation_target)
        mutants = parse_single_mutations(
            df,
            resolved_label,
            wt_ab,
            wt_ag,
            resolved_wt_kd,
            better_direction,
            target,
        )
        if not mutants:
            raise ValueError(
                "未筛到优于 WT 的单突变；请检查标签列 / better-direction / WT KD"
            )
    else:
        # 无野生型（或有序列但无 WT KD）：用一致性序列（若仍无序列）+ 亲和力 Top-N
        if not (wt_ab and wt_ag):
            selection_mode = "top-n-consensus"
            print(
                f"[mode] --input 无野生型 (source={wt_source})，"
                f"启用一致性序列 + Top-{top_n_singles}"
            )
            wt_ab, wt_ag = infer_wt_from_consensus(df)
        else:
            selection_mode = "top-n-with-wt-seq"
            print(
                f"[mode] 已有野生型序列 (source={wt_source}) 但无 WT KD，"
                f"启用 Top-{top_n_singles}（不与 WT KD 比较）"
            )
        resolved_wt_kd = None
        target = detect_mutation_target(df, wt_ab, wt_ag, mutation_target)
        mutants = select_top_affinity_singles(
            df,
            resolved_label,
            wt_ab,
            wt_ag,
            better_direction,
            top_n_singles,
            target,
        )
        if consensus_wt_out and selection_mode == "top-n-consensus":
            out_wt = Path(consensus_wt_out)
            out_wt.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        ab_col: wt_ab,
                        ag_col: wt_ag,
                        "mutation_id": "WT_CONSENSUS",
                        "source": "consensus_from_single_mutants",
                        "top_n_singles": top_n_singles,
                    }
                ]
            ).to_csv(out_wt, index=False)
            print(f"[consensus] wrote {out_wt}")

    if selected_singles_out:
        out_sel = Path(selected_singles_out)
        out_sel.parent.mkdir(parents=True, exist_ok=True)
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
                    "target": m.target,
                    "selection_mode": selection_mode,
                }
                for m in mutants
            ]
        ).to_csv(out_sel, index=False)

    n_ben = len(mutants)
    hi = min(max_order, n_ben) if max_order > 0 else n_ben
    lo = max(2, min_order)
    if max_combinations <= 0 and hi >= 12 and (hi - lo + 1) > 1:
        # 避免 C(30,k) 全阶穷举在构建阶段爆炸；须显式限制阶数或采样上限
        raise ValueError(
            f"增益单突变={n_ben} 且 max_order={max_order}、max_combinations=0，"
            f"组合空间过大。请设置 max-order（如 2~4）或 max-combinations（如 5000）"
        )

    combos = enumerate_combinations(
        mutants, min_order, max_order, max_combinations, seed
    )
    if not combos:
        raise ValueError(
            f"多突变组合数为 0（beneficial={n_ben}, "
            f"min_order={min_order}, max_order={max_order}）"
        )

    out_df = combinations_to_dataframe(combos, wt_ab, wt_ag, target)
    for col in ("antibody_seq", "antigen_seq"):
        if col not in out_df.columns or out_df[col].isna().any():
            raise ValueError(f"组合输出缺少有效列 {col}")
    mut_col = "antibody_seq" if target == "antibody" else "antigen_seq"
    fixed_col = "antigen_seq" if target == "antibody" else "antibody_seq"
    fixed_wt = wt_ag if target == "antibody" else wt_ab
    if out_df[mut_col].map(len).nunique() != 1:
        raise ValueError(f"组合输出 {mut_col} 长度不一致")
    if not (out_df[fixed_col] == fixed_wt).all():
        raise ValueError(f"组合输出固定链 {fixed_col} 应全等于野生型")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    summary = {
        "n_input_rows": int(len(df)),
        "wt_source": wt_source,
        "selection_mode": selection_mode,
        "mutation_target": target,
        "wt_kd": resolved_wt_kd,
        "label_col": resolved_label,
        "better_direction": better_direction,
        "top_n_singles": top_n_singles if selection_mode == "top-n-consensus" else None,
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
    p.add_argument(
        "--train-path",
        default="",
        help="可选：扩展训练 CSV（本脚本不校验、不用于 WT/突变链判定；仅下游 train 使用）",
    )
    p.add_argument(
        "--val-path",
        default="",
        help="可选：验证 CSV（本脚本不校验、不参与组合构建）",
    )
    p.add_argument(
        "--top-n-singles",
        type=int,
        default=30,
        help="无野生型时，按亲和力选取位点去重后的 Top-N 单突变作为组合备选",
    )
    p.add_argument(
        "--consensus-wt-out",
        default="",
        help="无野生型时，将一致性野生型序列写出到该路径",
    )
    p.add_argument(
        "--mutation-target",
        choices=["auto", "antibody", "antigen"],
        default="auto",
        help="突变施加链：auto 按序列自动判定；antibody/antigen 显式指定",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI 入口：校验数据并构建多突变库。"""
    args = parse_args(argv)
    try:
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
            train_path=args.train_path or None,
            val_path=args.val_path or None,
            top_n_singles=args.top_n_singles,
            consensus_wt_out=args.consensus_wt_out or None,
            mutation_target=args.mutation_target,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: 数据校验/组合失败: {exc}", file=sys.stderr)
        return 1
    print("=== build_multimutant_library summary ===")
    for k, v in summary.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
