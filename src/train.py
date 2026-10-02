"""
모델 학습 · 검증 · 평가 파이프라인 (설계 문서 5.2 ~ 5.5 구현)

- Task      : Classification (cycle_life ≥ 550 → Long=1 / Short=0), 지표 F1-Score · Accuracy
- 모델링    : log10(cycle_life) 회귀 → 예측 수명 550 기준 판정  (회귀는 분류를 위한 학습 수단)
- 데이터 분할
    Train (B1 CV)       : B1 학습 구간에서 충전 방식 단위 GroupKFold 평균
    Valid (B1 Hold-out) : B1 에서 충전 방식 단위로 약 20% 분리
    Test  (B2)          : 주 테스트,   Test (B3) : 추가 검증
- 누수 방지 : 결측 대체값·스케일러·하이퍼파라미터·피처 조합·판정 기준(550)은 모두 B1 안에서만 결정
  (결측 : B2 6셀은 초기 내부저항 미측정(IR=0) → B1 학습 데이터 중앙값으로 대체)

실행 : python -m src.train   (results/ 에 결과 저장)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import (CORE, FEATURE_SETS, W5_BASELINE, build_feature_table, select_b1_only)
from src.preprocess import ROOT, THRESHOLD, clean, load_raw

warnings.filterwarnings('ignore')

SEED = 42
N_SPLITS = 5
HOLDOUT_SIZE = 0.2
TARGET_ACC = 0.951          # 원논문 분류 성능 (1 − 4.9%)
TARGET_ERR = 0.049
SIMPLER_MARGIN = 0.5        # 피처 세트 선택 : CV MAPE 차이가 0.5%p 미만이면 피처 수가 적은 쪽을 택함
RESULTS = ROOT / 'results'
GROUP = 'policy_base'       # 충전 방식 단위 분할

EN_GRID = [dict(alpha=a, l1_ratio=r) for a in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3] for r in [0.1, 0.5, 0.9]]


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
class RegThreshold:
    """log10(수명) 회귀 → 예측 수명 ≥ 550 이면 Long. 중도 종료 셀은 회귀 학습에서 제외"""

    def __init__(self, kind, features):
        self.kind, self.features = kind, list(features)
        self.params = {}

    def _estimator(self, params):
        if self.kind == 'elasticnet':
            return ElasticNet(max_iter=50000, random_state=SEED, **params)
        if self.kind == 'linear':
            return LinearRegression()
        if self.kind == 'rf':
            return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED)
        raise ValueError(self.kind)

    def _tune(self, tr):
        """ElasticNet 하이퍼파라미터 : 학습 데이터 안의 GroupKFold(log MAE 최소)로만 선택"""
        n_groups = tr[GROUP].nunique()
        inner = GroupKFold(n_splits=min(4, n_groups))
        best, best_err = None, np.inf
        for params in EN_GRID:
            errs = []
            for i_tr, i_va in inner.split(tr, groups=tr[GROUP]):
                a, b = tr.iloc[i_tr], tr.iloc[i_va]
                m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), self._estimator(params)).fit(a[self.features], a['log_life'])
                errs.append(np.mean(np.abs(m.predict(b[self.features]) - b['log_life'])))
            if np.mean(errs) < best_err:
                best, best_err = params, np.mean(errs)
        return best

    def fit(self, df):
        tr = df[~df['censored']]
        self.params = self._tune(tr) if self.kind == 'elasticnet' else {}
        self.model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), self._estimator(self.params))
        self.model.fit(tr[self.features], tr['log_life'])
        self.trained = True
        return self

    def predict(self, df):
        log_life = self.model.predict(df[self.features])
        life = 10 ** log_life
        return dict(life=life, label=(life >= THRESHOLD).astype(int), score=log_life)

    def coef(self):
        est = self.model[-1]
        if hasattr(est, 'coef_'):
            return pd.Series(est.coef_, index=self.features)
        return pd.Series(est.feature_importances_, index=self.features)


class LogisticDirect:
    """분류기 직접 학습 (비교용). 학습 데이터에 Short 가 없으면 학습 불가 → 다수 클래스(Long) 예측으로 대체"""

    def __init__(self, features):
        self.features = list(features)

    def fit(self, df):
        y = df['label']
        self.trained = y.nunique() == 2
        if self.trained:
            self.model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=5000))
            self.model.fit(df[self.features], y)
        return self

    def predict(self, df):
        if not self.trained:
            n = len(df)
            return dict(life=None, label=np.ones(n, int), score=np.full(n, 0.5))
        proba = self.model.predict_proba(df[self.features])[:, 1]
        return dict(life=None, label=(proba >= 0.5).astype(int), score=proba)


class AllLong:
    """최저 기준선 : 모든 셀을 Long 으로 예측"""
    features = []

    def fit(self, df):
        self.trained = True
        return self

    def predict(self, df):
        n = len(df)
        return dict(life=None, label=np.ones(n, int), score=np.zeros(n))


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------
def metrics(df, pred):
    y, yp = df['label'].values, pred['label']
    out = dict(
        n=len(df), n_short=int((y == 0).sum()),
        f1=f1_score(y, yp, pos_label=1, zero_division=0),
        accuracy=accuracy_score(y, yp),
        macro_f1=f1_score(y, yp, labels=[0, 1], average='macro', zero_division=0),
        short_recall=recall_score(y, yp, pos_label=0, zero_division=0) if (y == 0).any() else np.nan,
        mape=np.nan, spearman=np.nan, auc=np.nan,
    )
    if pred['life'] is not None:
        unc = ~df['censored'].values                      # 중도 종료 셀은 수명이 하한값 → 오차 계산 제외
        life, true = pred['life'][unc], df['cycle_life'].values[unc]
        out['mape'] = np.mean(np.abs(life - true) / true) * 100
        out['spearman'] = stats.spearmanr(life, true).correlation
    if (y == 0).any() and (y == 1).any() and np.ptp(pred['score']) > 0:
        out['auc'] = roc_auc_score(y, pred['score'])
    return out


def split_b1(feat):
    """B1 을 충전 방식 단위로 학습(80%) / Hold-out(20%) 분리"""
    b1 = feat[feat['batch'] == 'B1'].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT_SIZE, random_state=SEED)
    i_tr, i_ho = next(gss.split(b1, groups=b1[GROUP]))
    return b1.iloc[i_tr].reset_index(drop=True), b1.iloc[i_ho].reset_index(drop=True)


def cv_evaluate(factory, train_df):
    """Train (B1 CV) : 충전 방식 단위 GroupKFold, fold 별 지표의 평균"""
    rows = []
    for k, (i_tr, i_va) in enumerate(GroupKFold(n_splits=N_SPLITS).split(train_df, groups=train_df[GROUP])):
        a, b = train_df.iloc[i_tr], train_df.iloc[i_va]
        m = factory().fit(a)
        r = metrics(b, m.predict(b))
        r.update(fold=k, train_short=int((a['label'] == 0).sum()), trained=m.trained)
        rows.append(r)
    return pd.DataFrame(rows)


def evaluate(name, feature_set, factory, b1_tr, b1_ho, b1_all, tests):
    """한 모델을 4개 구간(Train CV / Valid / Test B2 / Test B3)에서 평가"""
    folds = cv_evaluate(factory, b1_tr)
    rows = [dict(split='Train (B1 CV)', **folds.drop(columns=['fold', 'train_short', 'trained']).mean().to_dict(),
                 note=f"{int((~folds['trained']).sum())}개 fold 학습 불가 (학습 fold 에 Short 없음)"
                 if (~folds['trained']).any() else '')]
    m_tr = factory().fit(b1_tr)
    rows.append(dict(split='Valid (B1 Hold-out)', **metrics(b1_ho, m_tr.predict(b1_ho)), note=''))
    final = factory().fit(b1_all)                      # Test 는 B1 전체로 다시 학습한 모델로 평가
    preds = {}
    for b, df in tests.items():
        p = final.predict(df)
        preds[b] = p
        rows.append(dict(split=f'Test ({b})', **metrics(df, p), note=''))
    res = pd.DataFrame(rows)
    res.insert(0, 'feature_set', feature_set)
    res.insert(0, 'model', name)
    return res, final, folds, preds


def guide_table(res, test='B2'):
    """성능 리포팅 형식 (Classification) — F1-Score / Accuracy / 비고"""
    r = res.set_index('split')
    tr, va, te = r.loc['Train (B1 CV)'], r.loc['Valid (B1 Hold-out)'], r.loc[f'Test ({test})']
    rows = [
        ['Train (Batch 1 CV)', tr.f1, tr.accuracy, ''],
        ['Valid (Batch 1 Hold-out)', va.f1, va.accuracy, ''],
        [f'Test (Batch {test[-1]})', te.f1, te.accuracy, ''],
        ['Gap (Train-Valid)', tr.f1 - va.f1, tr.accuracy - va.accuracy, '(+) : 과적합 의심'],
        ['Gap (Valid-Test)', va.f1 - te.f1, va.accuracy - te.accuracy, '(+) : 배치간 일반화 저하 의심'],
        ['Gap (Target-Test)', np.nan, TARGET_ACC - te.accuracy, f'Target : Accuracy {TARGET_ACC * 100:.1f}%'],
    ]
    return pd.DataFrame(rows, columns=['구분', 'F1-Score', 'Accuracy', '비고'])


def guide_table_b3(res):
    """성능 리포팅 형식 (Batch 3 추가) — B2 결과와 나란히"""
    r = res.set_index('split')
    b2, b3 = r.loc['Test (B2)'], r.loc['Test (B3)']
    t = guide_table(res, 'B2')
    extra = pd.DataFrame([
        ['Test (Batch 3)', b3.f1, b3.accuracy, f'Short {int(b3.n_short)}개 → 수명 MAPE {b3.mape:.1f}%·순위 상관 {b3.spearman:.2f} 중심 해석'],
        ['Gap (Batch2-Batch3)', b2.f1 - b3.f1, b2.accuracy - b3.accuracy, 'Test 성능 간 비교'],
        ['Gap (Target-Test, Batch 3)', np.nan, TARGET_ACC - b3.accuracy, 'Batch 3 기준, 원논문 성능 비교 (B3 정확도는 "전부 Long" 과 같은 값 → 성능 근거 아님)'],
    ], columns=t.columns)
    return pd.concat([t, extra], ignore_index=True)


def recalibration_scenario(final, b2, ks=(3, 5, 10), n_rep=300):
    """
    운영 시나리오 (본 성능과 분리 보고) : 새 배치(B2)의 소량 라벨 k개로 판정 기준(절편)만 다시 맞춤
    - k개 셀에서 log(실제/예측) 평균만큼 예측을 이동 → 나머지 셀로 평가, n_rep 회 반복 평균
    """
    rng = np.random.default_rng(SEED)
    base = final.predict(b2)['score']
    true = b2['log_life'].values
    m0 = metrics(b2, final.predict(b2))
    rows = [dict(k=0, accuracy=m0['accuracy'], macro_f1=m0['macro_f1'], f1=m0['f1'])]
    for k in ks:
        acc, mf1, f1 = [], [], []
        for _ in range(n_rep):
            idx = rng.choice(len(b2), k, replace=False)
            rest = np.setdiff1d(np.arange(len(b2)), idx)
            shift = np.mean(true[idx] - base[idx])
            life = 10 ** (base[rest] + shift)
            sub = b2.iloc[rest]
            m = metrics(sub, dict(life=life, label=(life >= THRESHOLD).astype(int), score=base[rest] + shift))
            acc.append(m['accuracy']); mf1.append(m['macro_f1']); f1.append(m['f1'])
        rows.append(dict(k=k, accuracy=np.mean(acc), accuracy_std=np.std(acc),
                         macro_f1=np.mean(mf1), f1=np.mean(f1)))
    return pd.DataFrame(rows)


def error_table(df, pred):
    """테스트 배치 셀별 예측 결과 (오류 분석용)"""
    out = df[['cell', 'batch', 'policy', 'newstruct', 'censored', 'cycle_life', 'label', 'w100_dQ_logvar']].copy()
    out['pred_life'] = np.round(pred['life']).astype(int)
    out['pred_label'] = pred['label']
    out['correct'] = out['label'] == out['pred_label']
    out['ratio_pred_true'] = out['pred_life'] / out['cycle_life']
    return out


# ---------------------------------------------------------------------------
# 전체 실행
# ---------------------------------------------------------------------------
def run_all(feat=None, save=True, verbose=True):
    if feat is None:
        cells, meta = clean(load_raw(verbose=verbose))
        feat = build_feature_table(cells, meta)
    b1_tr, b1_ho = split_b1(feat)
    b1_all = feat[feat['batch'] == 'B1'].reset_index(drop=True)
    tests = {b: feat[feat['batch'] == b].reset_index(drop=True) for b in ['B2', 'B3']}

    # 1) 피처 세트 선택 : B1 학습 구간 CV(MAPE) 로만 결정
    sel_rows = []
    for name, fs in FEATURE_SETS.items():
        f = cv_evaluate(lambda fs=fs: RegThreshold('elasticnet', fs), b1_tr)
        sel_rows.append(dict(feature_set=name, n_features=len(fs), cv_mape=f['mape'].mean(), cv_mape_std=f['mape'].std(),
                             cv_spearman=f['spearman'].mean()))
    sel = pd.DataFrame(sel_rows).sort_values(['cv_mape'])
    best = sel.iloc[0]
    simpler = sel[(sel['cv_mape'] - best['cv_mape'] < SIMPLER_MARGIN)].sort_values(['n_features', 'cv_mape']).iloc[0]
    chosen_name = simpler['feature_set']
    chosen = FEATURE_SETS[chosen_name]

    # 2) 누수 점검용 B1 단독 선별 세트
    b1_only, b1_rho = select_b1_only(b1_tr)

    # 3) 후보 모델 평가 (설계 문서 [표 11] + 5.2 / 5.5 비교 실험)
    experiments = [
        ('ElasticNet → 550', chosen_name, lambda: RegThreshold('elasticnet', chosen)),
        ('단일 피처 선형회귀 → 550', 'core', lambda: RegThreshold('linear', CORE)),
        ('로지스틱 회귀 (balanced)', chosen_name, lambda: LogisticDirect(chosen)),
        ('RandomForest → 550', chosen_name, lambda: RegThreshold('rf', chosen)),
        ('전부 Long', '-', lambda: AllLong()),
        ('ElasticNet → 550 [B1 단독 선별]', 'b1_only', lambda: RegThreshold('elasticnet', b1_only)),
        ('ElasticNet → 550 [5사이클 기준선]', 'w5_baseline', lambda: RegThreshold('elasticnet', W5_BASELINE)),
    ]
    all_res, finals, folds, preds = [], {}, {}, {}
    for name, fs, factory in experiments:
        res, final, fold_df, pred = evaluate(name, fs, factory, b1_tr, b1_ho, b1_all, tests)
        all_res.append(res); finals[name] = final; folds[name] = fold_df; preds[name] = pred
    perf = pd.concat(all_res, ignore_index=True)

    main = 'ElasticNet → 550'
    main_res = perf[perf['model'] == main]
    out = dict(
        feat=feat, b1_tr=b1_tr, b1_ho=b1_ho, tests=tests,
        selection=sel, chosen_name=chosen_name, chosen=chosen, b1_only=b1_only, b1_rho=b1_rho,
        perf=perf, finals=finals, folds=folds, preds=preds,
        guide=guide_table(main_res), guide_b3=guide_table_b3(main_res),
        recal=recalibration_scenario(finals[main], tests['B2']),
        errors={b: error_table(tests[b], preds[main][b]) for b in tests},
        coef=finals[main].coef(), params=finals[main].params,
    )
    if save:
        save_results(out)
    return out


def save_results(out):
    RESULTS.mkdir(exist_ok=True)
    out['perf'].round(4).to_csv(RESULTS / 'model_performance.csv', index=False, encoding='utf-8-sig')
    out['guide'].round(4).to_csv(RESULTS / 'performance_report.csv', index=False, encoding='utf-8-sig')
    out['guide_b3'].round(4).to_csv(RESULTS / 'performance_report_b3.csv', index=False, encoding='utf-8-sig')
    out['selection'].round(4).to_csv(RESULTS / 'feature_set_selection.csv', index=False, encoding='utf-8-sig')
    out['recal'].round(4).to_csv(RESULTS / 'recalibration_scenario.csv', index=False, encoding='utf-8-sig')
    pd.concat(out['errors'].values()).to_csv(RESULTS / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    with open(RESULTS / 'final_model.json', 'w', encoding='utf-8') as fp:
        json.dump(dict(model='ElasticNet (log10 cycle_life) → 550 threshold', feature_set=out['chosen_name'],
                       features=out['chosen'], params=out['params'], coef=out['coef'].round(5).to_dict(),
                       b1_only_features=out['b1_only'], seed=SEED), fp, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    o = run_all()
    pd.set_option('display.width', 200)
    print('선택된 피처 세트 :', o['chosen_name'], o['chosen'])
    print(o['selection'].round(3).to_string(index=False))
    print(o['guide'].round(3).to_string(index=False))
    print(o['perf'][['model', 'split', 'f1', 'accuracy', 'macro_f1', 'mape', 'auc']].round(3).to_string(index=False))
