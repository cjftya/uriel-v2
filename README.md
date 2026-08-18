# Uriel v2

재현 가능한 한국 로또 6/45 시드 실험과 범용 확률 알고리즘 성능 예측 연구를 함께 관리하는 Python 프로젝트입니다. 두 연구 영역은 패키지와 출력 경로를 분리하며, UI 없이 CLI, 로그, Parquet/CSV/JSON 결과에 집중합니다.

현재 v0.7의 범위는 다음과 같습니다.

- `lotto.xlsx` 구조 자동 인식 및 데이터 검증
- Python 버전과 무관하게 같은 결과를 내는 SplitMix64 번호 생성기
- 미래 당첨번호를 사용하지 않는 3가지 시드 전략
- Process worker 기반 walk-forward 평가
- 실제 정답 번호로 시드 범위를 탐색하는 역산 진단
- 여러 회차를 같은 고정 seed budget으로 탐색하는 `reverse-batch`
- 회차별 Top-K, 4+/5+/6 seed, seed bucket, reconstruction curve 저장
- exact hypergeometric 및 고정 시드 Monte Carlo random baseline
- 다중 좌표 seed landscape를 다음 회차로 운반하는 `seed-field` walk-forward 실험
- 정답 없이 다음 회차 후보를 저장하는 `seed-field-predict`
- 8,145,060개 조합 공간의 Combinadic Rank Dynamics walk-forward 실험
- reverse 4+/5+/6-hit landscape를 이용한 Seed Basin/Attractor 실험
- 동일 candidate budget의 matched random 기준선, permutation test, bootstrap CI
- CSV/JSON 결과와 PNG 진단 그래프
- 6개 multi-view state와 exact/Derivative DTW 기반의 불규칙 recurrence motif 탐색
- KMeans·GMM·HDBSCAN regime과 variable-length transition motif 검증
- Historical 설정 고정, Development 무재튜닝, 4개 surrogate와 10,000회 random candidate baseline
- 콘솔 로그와 실행별 `outputs/` 결과 보존
- 로또 코드와 분리된 범용 확률 알고리즘 실험 패키지
- 공통 Problem/Algorithm/Budget/Run/Trace schema와 확장 JSON field
- PCG64 seed 격리, process worker, checkpoint/resume, Parquet dataset
- IID Monte Carlo와 Random Search 기반 Phase 1 smoke pilot
- 5%/10%/20% early-trajectory feature와 자동 데이터 품질 검사

> 역산 탐색은 이미 알려진 정답을 사용합니다. 시드 공간의 구조와 근접도를 살피는 진단 실험이지, 그 자체가 미래 회차 예측은 아닙니다.

## 범용 확률 알고리즘 연구

`uriel_v2.probabilistic_lab`은 기존 로또 실험과 코드·CLI·출력 경로를 분리한 범용 실험 기반입니다. 문제 구조, 랜덤 메커니즘, 예산, 초기 trajectory로 최종 quality/runtime/failure 분포를 예측하는 장기 연구의 Phase 1을 담당합니다.

```bash
python -m uriel_v2.probabilistic_lab pilot \
  --instances-per-family 4 \
  --seeds 3 \
  --workers auto
```

현재 smoke pilot은 Gaussian/Student-t/Mixture mean estimation과 Sphere/Rastrigin/Rosenbrock optimization을 생성하고, 각 domain에 IID Monte Carlo 또는 Random Search를 적용합니다. 생성되는 주요 파일은 다음과 같습니다.

| 파일 | 내용 |
|---|---|
| `data/problems/problem_metadata.parquet` | 고정 공통 problem schema |
| `data/runs/runs.parquet` | quality, runtime, failure, first-passage 결과 |
| `data/traces/common/trace_common.parquet` | 1/2/5/10/20/50/100% 공통 trace |
| `data/features/trajectory_features.parquet` | 5/10/20% early-trajectory feature |
| `checkpoint.jsonl` | run 단위 중간 복구 기록 |
| `validation.json` | duplicate, missing trace, NaN/Inf 등 품질 검사 |
| `summary.json` | 알고리즘·problem family별 smoke 결과 |

생성한 데이터셋은 다시 검사할 수 있습니다.

```bash
python -m uriel_v2.probabilistic_lab validate \
  artifacts/probabilistic/RUN_DIRECTORY
```

## 설치

Python 3.11 이상을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## 빠른 실행

### 1. 데이터 확인

```bash
python -m uriel_v2 inspect --data lotto.xlsx --rows 10
```

회차 수, 범위, 최신 번호와 최근 10개 회차를 콘솔과 `data-summary.json`에서 확인할 수 있습니다.

### 2. 다음 회차 시드 생성

```bash
python -m uriel_v2 predict --data lotto.xlsx
```

특정 회차와 여러 variant를 만들 수도 있습니다.

```bash
python -m uriel_v2 predict --round 1236 --candidates 10 --history-window 64
```

### 3. Walk-forward 평가

```bash
python -m uriel_v2 evaluate --start-round 1044 --end-round 1235 --workers auto
```

`evaluation.csv`에는 회차별 시드, 생성 번호, 정답, 적중 수와 다음 거리 지표가 저장됩니다.

- `set_distance`: 두 6개 번호 집합의 대칭 차이 크기. 완전 일치 0, 공통 번호가 없으면 12
- `positional_mae`: 정렬된 6개 번호를 위치별로 비교한 평균 절대 거리
- `signed_bias`: 정답보다 위/아래 번호로 치우친 평균 방향
- `delta_1`~`delta_6`: 각 정렬 위치의 `생성 번호 - 정답 번호`

`--candidates 1`은 실제 단일 시드 성능입니다. 2 이상이면 여러 variant 중 **정답 기준 최고 결과(oracle best)**를 기록하므로 후보 공간의 기회 품질 진단으로만 해석해야 합니다.

### 4. 정답 기반 역산 시드 탐색

```bash
python -m uriel_v2 reverse --round 1235 --seed-start 0 --seed-end 1000000 --workers auto
```

번호를 직접 지정할 수도 있습니다.

```bash
python -m uriel_v2 reverse \
  --numbers 6,7,11,15,39,43 \
  --seed-start 0 \
  --seed-end 10000000 \
  --min-hits 5 \
  --chunk-size 50000 \
  --workers 8
```

탐색 구간은 `[seed-start, seed-end)`입니다. 작업은 청크 단위로 여러 프로세스에 분산되며, 진행률과 속도는 로그에 계속 표시됩니다. `Ctrl+C`로 중단해도 해당 실행의 로그는 남습니다.

### 5. 여러 회차의 reverse reconstruction 데이터셋 구축

Stage A의 고정 조건(1044~1235회, 회차별 `[0, 1,000,000)`)은 다음과 같이 실행합니다.

```bash
python -m uriel_v2 reverse-batch \
  --start-round 1044 \
  --end-round 1235 \
  --seed-start 0 \
  --seed-end 1000000 \
  --top-k 100 \
  --min-hits 4 \
  --chunk-size 25000 \
  --bucket-size 100000 \
  --workers auto
```

모든 회차는 같은 seed 범위를 탐색합니다. 이 명령은 회차별로 진행률, 현재 최고 hit/seed, 5-hit·6-hit 수, 처리 속도와 ETA를 로그에 남깁니다. 결과는 `outputs/reverse-dataset/YYYYMMDD-HHMMSS-reverse-batch/`에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `reverse-rounds.csv` | 회차별 최고 결과, hit 분포, 실행 시간과 처리량 |
| `reverse-top-k.csv` | 회차별 결정적 정렬 기준의 Top-K seed |
| `reverse-hit-seeds.csv` | `min-hits` 이상인 모든 seed와 거리 지표 |
| `reverse-seed-buckets.csv` | seed bucket별 4/5/6-hit 수와 최고 결과 |
| `reverse-reconstruction-curve.csv` | 10K·100K·전체 budget의 회차별 최고 결과 |
| `reverse-summary.json` | 전체 reconstruction, random baseline, landscape 요약 |
| `curve-*.svg` | budget별 reconstruction curve 4종 |

Top-K 정렬은 `hits 내림차순 → positional_mae 오름차순 → set_distance 오름차순 → seed 오름차순`으로 고정됩니다. 보너스 번호는 이 reverse reconstruction 점수에 사용하지 않습니다.

### 6. Seed Field Transport 검증

`reverse-hit-seeds.csv`의 4+/5+/6-hit seed를 12개 좌표계로 투영하고, 직전 field의 지속·EWMA·유사 회차·원형 이동·XOR 이동과 균등 ensemble을 walk-forward로 평가합니다.

```bash
python -m uriel_v2 seed-field \
  --landscape outputs/reverse-dataset/HISTORICAL/reverse-hit-seeds.csv \
  --landscape outputs/reverse-dataset/DEVELOPMENT/reverse-hit-seeds.csv \
  --start-round 1044 \
  --end-round 1235 \
  --cohort Development
```

평가 결과에는 모델·회차·예산별 집계와 실제 Top-10 seed가 함께 저장됩니다.

| 파일 | 내용 |
|---|---|
| `seed-field-evaluation.csv` | 10·100·1K·10K budget의 exact 4/5/6-hit 수와 Top-10 적중 분포 |
| `seed-field-top10-seeds.csv` | 모델별 실제 Top-10 seed, field 점수, 생성 번호와 적중 수 |
| `seed-field-summary.json` | random null 대비 lift와 10,000회 Monte Carlo 단측 p-value |
| `uriel.log` | 16회 간격 진행률과 최종 ensemble 요약 |

다음 회차의 후보만 만들 때는 정답 데이터가 필요하지 않습니다.

```bash
python -m uriel_v2 seed-field-predict \
  --landscape outputs/reverse-dataset/HISTORICAL/reverse-hit-seeds.csv \
  --landscape outputs/reverse-dataset/DEVELOPMENT/reverse-hit-seeds.csv \
  --round 1236 \
  --top-k 100
```

`seed-field-candidates.csv`에는 모델별 순위, seed, field 점수와 생성 번호가 저장됩니다. 이 명령은 후보를 재현하기 위한 것으로, 로또 추첨의 예측 가능성을 전제하지 않습니다.

### 7. Combinadic Rank Dynamics

당첨번호 조합을 `0..8,145,059`의 유일한 lexicographic rank로 바꾸고 delta continuation, 과거 delta pattern, modulo consensus, nearest state를 독립적으로 계산합니다. 각 target은 반드시 직전 회차까지만 사용합니다.

```bash
python -m uriel_v2 combinadic-rank \
  --data lotto.xlsx \
  --start-round 852 \
  --end-round 1235 \
  --minimum-history 200 \
  --split-round 1044 \
  --seed 20260814
```

결과는 `artifacts/combinadic/<run>/` 아래의 `ranks.csv`, `predictions.csv`, `walk_forward.csv`, `metrics.json`과 PNG 그래프로 저장됩니다. Top-10/100/1,000/10,000은 예측 rank 중심의 동일한 후보창을 사용하며 random도 같은 구조의 후보창으로 비교합니다.

### 8. Reverse Seed Basin / Seed Attractor

`reverse-batch`가 만든 `reverse-hit-seeds.csv`를 재사용해 회차별 weighted center, width, 4+/5+ density, entropy, asymmetry와 exact seed 주변의 추가 고적중 밀도를 계산합니다. 예측 target 회차의 landscape는 forecast가 끝난 뒤 거리와 번호 적중 채점에만 사용합니다.

```bash
python -m uriel_v2 seed-basin \
  --data lotto.xlsx \
  --landscape outputs/reverse-dataset/HISTORICAL/reverse-hit-seeds.csv \
  --landscape outputs/reverse-dataset/DEVELOPMENT/reverse-hit-seeds.csv \
  --start-round 852 \
  --end-round 1235 \
  --minimum-history 32 \
  --split-round 1044 \
  --seed 20260814
```

결과는 `artifacts/seed_basin/<run>/` 아래의 `exact_seeds.csv`, `basin_summary.csv`, `basin_predictions.csv`, `walk_forward.csv`, `metrics.json`과 PNG 그래프로 저장됩니다. Stage A의 대규모 탐색은 기존 process worker와 checkpoint를 그대로 사용하고, 이 단계는 저장된 landscape를 빠르게 재평가합니다.

### 9. 두 실험 비교

```bash
python -m uriel_v2 compare-experiments \
  --combinadic-metrics artifacts/combinadic/RUN/metrics.json \
  --seed-basin-metrics artifacts/seed_basin/RUN/metrics.json
```

`artifacts/comparison/<run>/`에 candidate budget별 비교표와 `SUCCESS`, `WEAK SIGNAL`, `NO SIGNAL` 판정을 저장합니다. 두 알고리즘 중 하나라도 `SUCCESS`일 때만 Hybrid가 허용됩니다.

### 10. Irregular Recurrence Motif

각 회차를 raw number, 7×7 grid, circle, distribution, inter-round transition, local context의 6개 view로 분리하고, 길이가 다른 trajectory를 exact DTW와 Derivative-DTW로 비교합니다. 세 가지 사전등록 설정은 Historical에서만 비교하며 선택된 설정과 confidence 임계값을 Development에 그대로 적용합니다.

```bash
python -m uriel_v2 irregular-motif \
  --data lotto.xlsx \
  --start-round 852 \
  --end-round 1235 \
  --split-round 1044 \
  --workers 8 \
  --seed 20260814 \
  --output artifacts
```

중단된 선택 완료 이후 walk-forward는 이전 실행의 `checkpoint.jsonl` 또는 실행 디렉터리를 지정해 이어갈 수 있습니다.

```bash
python -m uriel_v2 irregular-motif ... --resume-from artifacts/motif/PREVIOUS_RUN
```

결과는 `artifacts/motif/<run>/`에 Parquet feature cache, recurrence candidates, 회차별 candidate ranking, opportunity subset, metrics와 recurrence/entropy/calibration 그래프로 저장됩니다.

### 11. Regime-Switching + Motif Transition

KMeans/GMM의 `K=4,6,8,12,16`과 HDBSCAN 두 설정을 Historical calibration에서 비교합니다. 고정된 설정으로 Regime-only와 variable-length transition motif를 각각 walk-forward 평가하며, 한 회차의 soft membership도 보존합니다.

```bash
python -m uriel_v2 regime-motif \
  --data lotto.xlsx \
  --start-round 852 \
  --end-round 1235 \
  --split-round 1044 \
  --workers 8 \
  --seed 20260814 \
  --output artifacts
```

Regime 실행도 회차별 prediction·match payload를 flush하므로 `--resume-from artifacts/regime/PREVIOUS_RUN`으로 이어갈 수 있습니다.

최종 비교는 두 실행의 `metrics.json`을 사용합니다.

```bash
python -m uriel_v2 motif-compare \
  --motif-metrics artifacts/motif/RUN/metrics.json \
  --regime-metrics artifacts/regime/RUN/metrics.json \
  --output artifacts
```

Hybrid(`C`)는 두 엔진이 모두 `SUCCESS`일 때만 허용됩니다. `WEAK SIGNAL`은 Locked/Blind 개방 조건이 아닙니다.

### 12. Opportunity Mechanism Analysis

동결된 `multiview_long` 실행을 다시 튜닝하지 않고, opportunity와 non-opportunity, 성공(4+)과 실패(<4)의 retrieval·cross-view·follow-up consensus·candidate 구조를 비교합니다. 최대 5개 Stage 2 규칙은 Historical에서만 정의·선택하고 Development에 그대로 적용합니다.

```bash
python -m uriel_v2 opportunity-analysis \
  --data lotto.xlsx \
  --motif-run artifacts/motif/RUN \
  --start-round 852 \
  --end-round 1235 \
  --split-round 1044 \
  --seed 20260814 \
  --output artifacts
```

결과는 `artifacts/opportunity_analysis/<run>/`에 opportunity feature/label, 성공·실패 통계, view ablation·single-view·pair diagnostic, motif family, second-order motif, Stage 2 rule, candidate funnel, missing winner, false positive, metrics와 PNG 그래프로 저장됩니다. View 진단은 target label을 사용하지 않고 기존 Top-40 match pool의 support vector만 재가중하며, Locked 660–851과 Additional Blind 468–659는 평가 target으로 열지 않습니다.

### 13. Top30 Broad-Area Retrieval

동결된 `multiview_long`과 Stage 1 confidence threshold를 그대로 사용해 Top30 broad-area retrieval을 별도 사전등록 실험으로 검증합니다. Seen 852–1235 재현이 완료돼야 `preregistration.json`을 기록하고 Locked 660–851을 한 번 평가합니다. Additional Blind 468–659는 Locked가 엄격한 SUCCESS 조건을 모두 통과할 때에만 CLI가 자동으로 엽니다.

```bash
python -m uriel_v2 top30-broad-retrieval \
  --data lotto.xlsx \
  --source-motif-run artifacts/motif/20260814-174919-irregular-motif \
  --source-opportunity-run artifacts/opportunity_analysis/20260815-124819-opportunity-analysis \
  --seen-start 852 --seen-end 1235 \
  --locked-start 660 --locked-end 851 \
  --blind-start 468 --blind-end 659 \
  --confidence-threshold 0.011722291804 \
  --candidate-size 30 \
  --seed 20260818 \
  --iterations 100000 \
  --workers 4 \
  --output artifacts
```

결과는 `artifacts/top30_broad_retrieval/<run>/`에 preregistration과 source validation, target별 Top30 prediction, block/cohort 통계, candidate funnel, surrogate, checkpoint, access log, metrics와 10개 PNG로 저장됩니다. `--force-blind` 같은 우회 옵션은 제공하지 않습니다.

## 시드 전략

| 전략 | 입력 | 목적 |
|---|---|---|
| `round` | 대상 회차, variant | 가장 단순한 기준선 |
| `history-digest` | 최근 N회 전체 번호 | 과거 상태 전체를 하나의 안정적 해시 시드로 압축 |
| `rolling-mix` | 빈도, 회차별 가중 합, 최근 간격 | 단순 통계 상태를 조합한 시드 |

모든 전략은 BLAKE2b로 64비트 시드를 만들고, 자체 SplitMix64 생성기로 1~45 중 6개를 선택합니다. 같은 데이터·옵션이면 운영체제와 Python 버전이 달라도 같은 시드와 번호가 나옵니다.

## 결과 구조

각 실행은 `outputs/YYYYMMDD-HHMMSS-command/` 아래에 저장됩니다.

```text
outputs/
└── 20260812-230000-evaluate/
    ├── uriel.log
    ├── evaluation.csv
    └── summary.json
```

긴 실행 중에는 콘솔과 `uriel.log`에서 진행 상황을 확인할 수 있습니다. 생성 결과 폴더는 Git에 포함되지 않습니다.

## 테스트

```bash
python -m unittest discover -s tests -v
# 또는 개발 의존성 설치 후: pytest
```
