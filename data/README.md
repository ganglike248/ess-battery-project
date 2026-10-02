# 데이터

## 원본 (Git 에 포함하지 않음, 약 8GB)
Kaggle : [Data-driven prediction of battery cycle life](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)

아래 3개 파일을 `data/raw/` 에 둠 (`archive/` 폴더에 두어도 자동 인식)

| 파일 | 배치 | 용도 |
|---|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 | 학습 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 | 테스트 (필수) |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 | 추가 검증 |

`2018-04-03_varcharge...` 파일은 다른 실험(가변 충전)이라 사용하지 않음

## 정제 규칙 (`src/preprocess.py`)
| 항목 | 처리 |
|---|---|
| 수명 정답이 없는 셀 (B2 VarCharge·SLOWCYCLE 8셀, B3 EOL 미도달 2셀) | 제외 |
| 원논문 노이즈 채널 (B3 c2, c37, c42, c43) | 제외 → B3 40셀 |
| 중도 종료 셀 (B1 10셀, 0.88Ah 도달 전 실험 종료) | 라벨 유지, 수명 회귀 학습에서 제외 |
| 초기 내부저항 0 (B2 6셀, 미측정) | 결측 처리 → B1 학습 데이터 중앙값으로 대체 |

최종 : B1 46셀 / B2 39셀 / B3 40셀

## 생성 파일
- `interim/` : 로딩 캐시 (Git 제외)
- `processed/features.csv` : 셀 단위 피처 테이블 (`02_feature_engineering.ipynb` 에서 생성)
- `processed/features_day1.csv` : Day 1 EDA 피처 테이블 (`01_EDA.ipynb` 에서 생성)
