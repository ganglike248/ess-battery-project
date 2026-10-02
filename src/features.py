"""
셀 단위 피처 생성 + 피처 세트 정의 (설계 문서 [표 2] 피처 구성, 5.1 Feature Engineering 구현)

피처는 같은 정의를 두 관측 기간으로 계산
- w5_   : 사이클 1~5   (ΔQ_5-4)     → 원논문 분류 설정, 5사이클 기준선에 사용
- w100_ : 사이클 1~100 (ΔQ_100-10)  → 본 프로젝트 입력
"""
import re

import numpy as np
import pandas as pd
from scipy import stats

POLICY_RE = re.compile(r'([\d.]+)C\((\d+)%\)-([\d.]+)C')

# ---------------------------------------------------------------------------
# 피처 세트 (설계 문서 5.1 / 5.4 / 5.5 와 1:1 대응)
# ---------------------------------------------------------------------------
CORE = ['w100_dQ_logvar']                       # 핵심 : ΔQ_100-10 분산(log)
AUX_CANDIDATES = ['w100_dQ_kurt', 'switch_soc']   # 보조 후보 : 첨도, 2단계 전환 시점 → B1 CV 로 채택 여부 결정

FEATURE_SETS = {
    'core': CORE,
    'core+kurt': CORE + ['w100_dQ_kurt'],
    'core+switch': CORE + ['switch_soc'],
    'core+kurt+switch': CORE + ['w100_dQ_kurt', 'switch_soc'],
}

# 5사이클 기준선 : 원논문 분류 모델과 같은 관측 기간(사이클 1~5)의 피처 구성
W5_BASELINE = ['w5_dQ_logvar', 'w5_dQ_logmin', 'QD2', 'w5_QDmax_m_QD2', 'w5_chargetime', 'w5_Tavg', 'w5_IR_min']

# EDA Q5 에서 검토한 후보 21개 (B1 단독 선별 세트를 고를 때의 후보 풀)
CANDIDATES_21 = ['w5_dQ_logvar', 'w5_dQ_logmin', 'w5_dQ_mean',
                 'w100_dQ_logvar', 'w100_dQ_logmin', 'w100_dQ_mean', 'w100_dQ_skew', 'w100_dQ_kurt',
                 'QD2', 'w5_QDmax_m_QD2', 'w100_QDmax_m_QD2', 'w100_fade_slope',
                 'w5_chargetime', 'w5_Tavg', 'w5_Tmax', 'w100_Tavg',
                 'IR2', 'w5_IR_min', 'w100_IR_delta', 'Cavg', 'C1']


def parse_policy(p):
    """'5.4C(50%)-3C' → C1 = 1단계 C-rate, switch_soc = 2단계 전환 SOC(%), C2 = 2단계 C-rate, Cavg = 0→80% 평균 C-rate"""
    m = POLICY_RE.search(p)
    if not m:
        return dict(C1=np.nan, switch_soc=np.nan, C2=np.nan, Cavg=np.nan)
    c1, q1, c2 = float(m.group(1)), float(m.group(2)) / 100, float(m.group(3))
    q1 = min(q1, 0.8)
    t = q1 / c1 + (0.8 - q1) / c2
    return dict(C1=c1, switch_soc=q1 * 100, C2=c2, Cavg=0.8 / t)


def dq_stats(dq, prefix):
    """ΔQ(V) 곡선 → 분산(log)·최솟값(log|·|)·평균·왜도·첨도"""
    dq = dq[~np.isnan(dq)]
    if dq.size == 0:
        return {f'{prefix}_{k}': np.nan for k in ['logvar', 'logmin', 'mean', 'skew', 'kurt']}
    return {
        f'{prefix}_logvar': np.log10(np.var(dq)),
        f'{prefix}_logmin': np.log10(np.abs(dq.min())),
        f'{prefix}_mean': dq.mean(),
        f'{prefix}_skew': stats.skew(dq),
        f'{prefix}_kurt': stats.kurtosis(dq),
    }


def window_feats(s, last, prefix):
    """summary 기반 피처 (사이클 1 ~ last)"""
    idx = np.arange(1, last + 1)
    qd = s['QDischarge'][idx]
    x = idx[1:]
    slope, _ = np.polyfit(x, s['QDischarge'][x], 1)
    ir = s['IR'][idx]
    ir_valid = ir[ir > 0]
    return {
        f'{prefix}_fade_slope': slope,
        f'{prefix}_chargetime': np.median(s['chargetime'][idx]),
        f'{prefix}_Tavg': s['Tavg'][idx].mean(),
        f'{prefix}_Tmax': s['Tmax'][idx].max(),
        f'{prefix}_IR_min': ir_valid.min() if ir_valid.size else np.nan,
        f'{prefix}_IR_delta': s['IR'][last] - s['IR'][2],
        f'{prefix}_QDmax_m_QD2': qd.max() - s['QDischarge'][2],
    }


def build_feature_table(cells, meta):
    """정제된 셀 → 피처 테이블 (메타 + 피처 + log_life)"""
    rows = []
    for c in cells:
        s, Q = c['summary'], c['Qdlin']
        ir2 = s['IR'][2]
        r = {'cell': c['cell'], 'QD2': s['QDischarge'][2], 'IR2': ir2 if ir2 > 0 else np.nan}   # IR=0 은 미측정 → 결측
        r.update(parse_policy(c['policy']))
        r.update(dq_stats(Q[5] - Q[4], 'w5_dQ'))
        r.update(dq_stats(Q[100] - Q[10], 'w100_dQ'))
        r.update(window_feats(s, 5, 'w5'))
        r.update(window_feats(s, 100, 'w100'))
        rows.append(r)
    feat = meta.merge(pd.DataFrame(rows), on='cell')
    feat['log_life'] = np.log10(feat['cycle_life'])
    return feat


def select_b1_only(train_df, candidates=CANDIDATES_21, k=4, corr_max=0.9):
    """
    테스트셋 누수 점검용 'B1 단독 선별' 세트 (설계 문서 Q5 한계 / 5.5)
    - B1 학습 데이터만 보고 |Spearman(피처, log 수명)| 이 큰 순으로 고르되,
      이미 고른 피처와 |상관| > corr_max 이면 건너뜀 (다중공선성 처리만 동일하게 적용)
    - B2·B3 정보는 전혀 사용하지 않음
    """
    rho = {c: stats.spearmanr(train_df[c], train_df['log_life'], nan_policy='omit').correlation for c in candidates}
    order = sorted(rho, key=lambda c: -abs(rho[c]))
    chosen = []
    for c in order:
        if all(abs(stats.spearmanr(train_df[c], train_df[o], nan_policy='omit').correlation) <= corr_max for o in chosen):
            chosen.append(c)
        if len(chosen) == k:
            break
    return chosen, pd.Series(rho).loc[order]
