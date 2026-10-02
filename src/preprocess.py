"""
데이터 로딩 · 정제 (설계 문서 [표 1] 데이터 정제 규칙 구현)

- .mat(HDF5) 파일에서 필요한 필드만 h5py 로 읽음 (cycle_life, policy, summary, Qdlin 사이클 1~100, Vdlin)
- 정제 규칙
    1. 수명 정답이 없는 셀 제외 (B2 VarCharge·SLOWCYCLE 8셀, B3 EOL 미도달 2셀)
    2. 원논문 노이즈 채널 제외 (B3 c2, c37, c42, c43) → B3 40셀
    3. 중도 종료(censored) 셀 표시 : 0.88Ah 도달 전 실험 종료 → 라벨은 유지, 회귀 학습에서는 제외
"""
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import medfilt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIRS = [ROOT / 'data' / 'raw', ROOT / 'archive']          # 둘 중 .mat 파일이 있는 곳을 사용
CACHE = ROOT / 'data' / 'interim' / 'cells.pkl'

THRESHOLD = 550        # Long / Short 기준 사이클 (원논문 분류 기준)
EOL_AH = 0.88          # 수명 종료 기준 = 공칭 1.1Ah 의 80%

BATCH_FILES = {
    'B1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',   # 학습
    'B2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',   # 주 테스트
    'B3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',   # 추가 검증 (선택)
}
BATCHES = list(BATCH_FILES)
PAPER_NOISY_B3 = ['B3c2', 'B3c37', 'B3c42', 'B3c43']


def find_data_dir():
    for d in DATA_DIRS:
        if all((d / fn).exists() for fn in BATCH_FILES.values()):
            return d
    raise FileNotFoundError(f'.mat 파일을 찾을 수 없음 : {[str(d) for d in DATA_DIRS]} (data/README.md 참고)')


def _str(f, ref):
    return ''.join(chr(c) for c in f[ref][()].flatten())


def load_batch(path, batch_name, max_cycle=100):
    """배치 파일 하나 → 셀별 dict 리스트 (Qdlin 은 row index = cycle 번호)"""
    cells = []
    with h5py.File(path, 'r') as f:
        b = f['batch']
        for i in range(b['summary'].shape[0]):
            s = f[b['summary'][i, 0]]
            cyc = f[b['cycles'][i, 0]]
            n_rows = cyc['Qdlin'].shape[0]
            qdlin = np.full((max_cycle + 1, 1000), np.nan)
            for j in range(1, min(max_cycle, n_rows - 1) + 1):
                q = f[cyc['Qdlin'][j, 0]][()]
                if q.size == 1000:
                    qdlin[j] = q.ravel()
            cells.append({
                'batch': batch_name,
                'cell': f'{batch_name}c{i}',
                'cycle_life': float(f[b['cycle_life'][i, 0]][()].item()),
                'policy': _str(f, b['policy_readable'][i, 0]),
                'summary': {k: s[k][0, :].astype(float) for k in s.keys()},
                'Qdlin': qdlin,
                'Vdlin': f[b['Vdlin'][i, 0]][()].ravel(),
            })
    return cells


def load_raw(use_cache=True, verbose=True):
    """세 배치 원본 로드 (캐시 사용)"""
    if use_cache and CACHE.exists():
        with open(CACHE, 'rb') as fp:
            raw = pickle.load(fp)
        if {c['batch'] for c in raw} == set(BATCHES):
            if verbose:
                print(f'캐시 로드 : {CACHE.relative_to(ROOT)}')
            return raw
    data_dir = find_data_dir()
    raw = []
    for name, fn in BATCH_FILES.items():
        if verbose:
            print(f'{name} 로딩 중 ... ({fn})')
        raw += load_batch(data_dir / fn, name)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, 'wb') as fp:
        pickle.dump(raw, fp)
    return raw


def build_meta(raw):
    """셀 단위 메타 정보 + 정제 플래그"""
    rows = []
    for c in raw:
        qd = c['summary']['QDischarge']
        rows.append({'cell': c['cell'], 'batch': c['batch'], 'policy': c['policy'],
                     'cycle_life': c['cycle_life'], 'last_QD': medfilt(qd, 5)[-3]})
    meta = pd.DataFrame(rows)
    meta['paper_noisy'] = meta['cell'].isin(PAPER_NOISY_B3)
    meta['excluded'] = meta['cycle_life'].isna() | meta['paper_noisy']
    meta['censored'] = (~meta['excluded']) & (meta['last_QD'] > EOL_AH + 0.01)
    meta['newstruct'] = meta['policy'].str.contains('newstructure')
    meta['policy_base'] = meta['policy'].str.replace('-newstructure', '', regex=False)
    return meta


def clean(raw):
    """정제 규칙 적용 → (분석 대상 셀 리스트, 메타 DataFrame)"""
    meta = build_meta(raw)
    keep = ~meta['excluded'].values
    cells = [c for c, k in zip(raw, keep) if k]
    meta = meta[keep].reset_index(drop=True)
    meta['cycle_life'] = meta['cycle_life'].astype(int)
    meta['label'] = (meta['cycle_life'] >= THRESHOLD).astype(int)    # 1 = Long, 0 = Short
    return cells, meta


if __name__ == '__main__':
    cells, meta = clean(load_raw())
    print(meta.groupby('batch').agg(cells=('cell', 'size'), long=('label', 'sum'),
                                    censored=('censored', 'sum')))
