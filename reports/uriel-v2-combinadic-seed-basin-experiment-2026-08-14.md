# Uriel v2 — Combinadic Rank Dynamics / Reverse Seed Basin 실험 보고서

- 작성일: 2026-08-14 (Asia/Seoul)
- 대상 저장소: `cjftya/uriel-v2`
- 데이터: `lotto.xlsx`, 1–1235회
- 평가 구간: Historical 852–1043회, Development 1044–1235회
- 실험 seed: `20260814`
- 최종 판정: **D. 둘 다 종료**

## Executive Summary

| 알고리즘 | 판정 | 핵심 근거 |
|---|---|---|
| Combinadic Rank Dynamics | **NO SIGNAL** | Ensemble rank 거리가 두 구간 모두 random과 유의한 차이가 없고, @1,000의 5+ 증가가 Development에서 재현되지 않음 |
| Reverse Seed Basin / Seed Attractor | **NO SIGNAL** | exact seed 주변 4+ 밀도가 random window보다 유의하게 높지 않고, center forecast와 candidate budget 성능이 Development에서 열세 |
| Optional Hybrid | **금지** | 두 독립 알고리즘 모두 승격 조건을 충족하지 못함 |

두 가설 모두 구현과 실행은 정상적으로 완료됐다. 그러나 새로운 상태 공간으로 변환한 뒤에도 다음 회차에 반복 가능한 구조는 확인되지 않았다.

Combinadic에서는 Historical @1,000의 5+가 5회로 matched random 2회보다 많았지만 Development에서는 1회 대 1회였다. Seed Basin에서는 Historical @10,000에 exact-6 2회와 5+ 49회가 관측됐지만 Development에서는 exact-6 0회, 5+ 49회 대 random 51회로 사라졌다. 어느 쪽도 out-of-sample 반복성을 만족하지 않는다.

따라서 이 결과를 이용한 Hybrid, 가중치 튜닝, Locked/Blind 개방은 하지 않는다.

## 1. 실험 범위와 재현성

| 항목 | 값 |
|---|---|
| Python | 3.12.13 |
| Uriel v2 | 0.3.0 |
| 최종 구현 commit | `cbcefe5a8a2eb68dec635aef5ade2d494f71df91` |
| 통합 실험 실행 HEAD | `46cb7b9ef6fde76a7a1733cb34779228e3f977fd` |
| Combinadic 실행 시간 | 84.21초 |
| Seed Basin 실행 시간 | 39.00초 |
| 원본 데이터 SHA-256 | `7efe5e232c7d4ed347b0377726686adad93ebe0581f3d03257cf4e2f83836db4` |
| Combinadic 평가 회차 | 384회 |
| Seed Basin 평가 회차 | 384회 |
| Random/permutation 반복 | 각 10,000회 |
| Locked 660–851 | **미사용** |
| Blind 468–659 | **미사용** |

통합 실험 실행 HEAD에는 최신 `main`의 Seed Field 작업과 이번 보고서 초안까지 포함돼 있다. 마지막 코드 변경 commit은 Combinadic 구성요소별 유의성 지표를 확장한 `85d8afd`다.

모든 target 회차 `t`의 forecast에는 `t-1`까지의 정보만 들어간다. target의 당첨번호와 reverse landscape는 forecast 생성 후 거리와 적중을 채점할 때만 사용했다.

## 2. 구현 내용

### Combinadic Rank Dynamics

- 6/45 조합을 lexicographic rank `0..8,145,059`로 양방향 변환
- rank, 1·2·3차 delta, circular distance, modulo 13종, bit/XOR feature 저장
- Delta median/weighted mean/trimmed mean
- 최근 3·5·8·13·21 delta pattern의 L1/normalized L2/cosine matching
- modulo consensus
- nearest historical state
- 구성요소와 ensemble 중심별 rank 거리 평가
- 예측 중심 주변 Top-10/100/1,000/10,000 rank window 평가
- 동일한 개수의 random center와 동일한 window 생성 규칙을 사용한 matched baseline

### Reverse Seed Basin / Seed Attractor

- 기존 Stage A에서 계산한 416회·4억 1,600만 seed 역산 결과 재사용
- 4+/5+/6-hit seed 578,427개로 회차별 basin 구성
- weighted center, P90-P10 width, 4+/5+ density, entropy, asymmetry 계산
- exact seed 주변 ±100/1,000/10,000/100,000의 추가 4+ seed 밀도 검정
- Center Delta, State Matching, Multi-Scale Center, Density Gradient와 ensemble forecast
- 예측 중심 주변 Top-10/100/1,000/10,000 seed candidate 평가
- Uriel SplitMix64 canonical generator 고정

Canonical generator fingerprint:

`eed2233e2b100f6c32d78fa29c1867a8b4dd8e43673f3155c4db688c9425085e`

## 3. Algorithm A — Combinadic Rank Dynamics

### 3.1 Rank 거리

거리 effect는 `Algorithm - Random`이다. 음수면 알고리즘이 더 가깝다.

| Cohort | Algorithm mean circular | Random mean circular | Effect | Bootstrap 95% CI | Paired p |
|---|---:|---:|---:|---:|---:|
| Historical | 2,019,083 | 1,975,878 | +43,205 | [-203,095, 286,981] | 0.638 |
| Development | 2,087,017 | 2,112,744 | -25,727 | [-273,119, 213,355] | 0.423 |

Historical에서는 random보다 멀었고 Development에서는 1.2% 가까웠지만, 두 차이 모두 신뢰구간이 0을 넓게 포함한다. 방향도 서로 반대다.

### 3.2 구성요소별 확인

Pattern Matching은 두 구간 모두 평균상 random보다 가까웠기 때문에 별도로 확인했다.

| Component | Historical effect / p | Development effect / p | 판정 |
|---|---:|---:|---|
| Delta Continuation | +32,957 / 0.605 | +113,260 / 0.833 | 열세 |
| Pattern Matching | -90,332 / 0.235 | -119,044 / 0.168 | 방향 일치, 유의하지 않음 |
| Modulo Consensus | +35,759 / 0.611 | -152,738 / 0.114 | 구간 불일치 |
| Nearest State | +161,549 / 0.908 | -22,767 / 0.428 | 구간 불일치 |
| Ensemble | +43,205 / 0.638 | -25,727 / 0.423 | 구간 불일치 |

Pattern Matching의 95% CI도 Historical `[-335,005, 158,809]`, Development `[-358,832, 122,503]`으로 모두 0을 포함한다. 흥미로운 평균 방향만으로 `WEAK SIGNAL`로 승격하지 않는다.

### 3.3 Candidate budget

각 셀의 `A/R`은 Algorithm/Matched Random이다.

| Cohort | Budget | Mean Max Hit A/R | 4+ A/R | 5+ A/R | 6 A/R | Paired p |
|---|---:|---:|---:|---:|---:|---:|
| Historical | 100 | 2.552 / 2.583 | 13 / 12 | 1 / 1 | 0 / 0 | 0.717 |
| Historical | 1,000 | 3.177 / 3.146 | 53 / 53 | 5 / 2 | 0 / 0 | 0.343 |
| Historical | 10,000 | 3.781 / 3.859 | 134 / 142 | 16 / 24 | 0 / 0 | 0.912 |
| Development | 100 | 2.479 / 2.500 | 12 / 10 | 0 / 0 | 0 / 0 | 0.646 |
| Development | 1,000 | 3.099 / 3.130 | 45 / 45 | 1 / 1 | 0 / 0 | 0.729 |
| Development | 10,000 | 3.750 / 3.766 | 128 / 131 | 20 / 17 | 0 / 0 | 0.627 |

Historical @1,000의 5+ `5 대 2`는 Development에서 `1 대 1`로 사라졌다. 반대로 Development @10,000의 5+는 20 대 17이지만 Historical에서는 16 대 24로 열세다.

정답 rank가 예측 window 자체에 들어온 사례는 모든 budget에서 0회였다. 6-hit도 0회였다.

### 3.4 Combinadic 판정

**NO SIGNAL**

- 조합 ↔ rank 변환과 모든 feature는 결정적이고 정상적으로 동작한다.
- Pattern Matching에 약한 평균 방향은 있으나 통계적으로 구분되지 않는다.
- rank 거리, 5+ 기회, exact rank window hit 어느 것도 두 구간에서 재현되지 않는다.
- 현 설정의 delta, modulo, bit/XOR, nearest state를 추가 튜닝하지 않는다.

## 4. Algorithm B — Reverse Seed Basin / Seed Attractor

### 4.1 Basin이 실제로 존재하는가

416회에서 exact-6 seed는 50개였다. 아래 표는 exact seed 자체를 제외한 주변 4+ seed 수와 같은 회차의 random center 주변 수를 비교한다.

| Radius | Exact 주변 평균 | Random 주변 평균 | Effect | Paired p |
|---:|---:|---:|---:|---:|
| ±100 | 0.34 | 0.34 | 0.00 | 0.572 |
| ±1,000 | 2.58 | 3.04 | -0.46 | 0.930 |
| ±10,000 | 28.02 | 26.80 | +1.22 | 0.146 |
| ±100,000 | 268.34 | 259.88 | +8.46 | 0.132 |

±10,000과 ±100,000에서 약한 양의 차이가 있지만 유의하지 않고, 작은 radius에서는 효과가 없다. exact seed 주변에 별도의 attractor basin이 존재한다고 볼 근거가 없다.

### 4.2 Basin continuity

| Metric | 값 |
|---|---:|
| Weighted center lag-1 correlation | 0.0195 |
| Center delta lag-1 correlation | -0.4971 |
| 평균 absolute center delta | 9,664 |
| 중앙 absolute center delta | 8,057 |

center level에는 연속성이 없다. delta의 음의 상관은 안정된 평균 주위의 noisy level을 차분할 때 생기는 기계적 반전 패턴과 일치하며, 다음 center를 맞히는 성능으로 이어지지 않았다.

### 4.3 Center prediction

거리 effect는 `Algorithm - Random`이며 음수면 더 가깝다.

| Target | Historical effect / p | Development effect / p | 판정 |
|---|---:|---:|---|
| Nearest 4+ seed | -7 / 0.431 | -65 / 0.035 | Development에서만 개선 |
| Nearest 5+ seed | +578 / 0.637 | +705 / 0.656 | 두 구간 모두 열세 |
| Nearest exact-6 seed | -33,831 / 0.166 (26회) | +252 / 0.509 (18회) | 재현 실패 |

Development의 nearest 4+ 개선은 nominal `p=0.035`지만 Historical에서 재현되지 않았고, 실제 목표인 nearest 5+와 candidate 성능으로 연결되지 않았다. 다중 metric 중 하나인 점까지 고려해 승격하지 않는다.

### 4.4 Candidate budget

| Cohort | Budget | Mean Max Hit A/R | 4+ A/R | 5+ A/R | 6 A/R | Paired p |
|---|---:|---:|---:|---:|---:|---:|
| Historical | 100 | 3.036 / 3.026 | 22 / 18 | 1 / 2 | 0 / 0 | 0.455 |
| Historical | 1,000 | 3.760 / 3.786 | 138 / 144 | 8 / 7 | 0 / 0 | 0.724 |
| Historical | 10,000 | 4.266 / 4.198 | 192 / 192 | 49 / 38 | 2 / 0 | 0.087 |
| Development | 100 | 3.005 / 3.130 | 16 / 36 | 0 / 1 | 0 / 0 | 0.998 |
| Development | 1,000 | 3.750 / 3.781 | 142 / 139 | 2 / 11 | 0 / 0 | 0.763 |
| Development | 10,000 | 4.255 / 4.266 | 192 / 192 | 49 / 51 | 0 / 0 | 0.643 |

Historical @10,000은 5+ 49 대 38, exact-6 2 대 0으로 가장 눈에 띄었다. 그러나 평균 Max Hit 검정은 `p=0.087`이고, Development에서는 5+ 49 대 51과 exact-6 0 대 0으로 사라졌다. 10,000개라는 큰 budget에서 한 구간에만 나타난 결과이므로 이상점으로 기록하되 신호로 인정하지 않는다.

Development @100은 4+가 16 대 36, @1,000은 5+가 2 대 11로 random보다 명확히 나빴다.

### 4.5 Seed Basin 판정

**NO SIGNAL**

- exact seed 주변 고적중 밀집은 random window와 구분되지 않는다.
- basin center의 level continuity가 없다.
- 5+와 exact seed 거리 개선이 독립 구간에서 반복되지 않는다.
- Historical @10,000 이상점은 Development에서 소멸했다.
- 현 center/gradient/state/multi-scale 조합은 종료한다.

## 5. 최종 비교와 결정

Primary budget @1,000 비교다. 각 알고리즘은 자기 상태 공간과 동일한 matched random을 사용하므로 random 열을 분리했다.

| Cohort | Metric | Combinadic Random | Combinadic | Basin Random | Seed Basin |
|---|---|---:|---:|---:|---:|
| Historical | Mean Max Hit | 3.146 | 3.177 | 3.786 | 3.760 |
| Historical | 4+ | 53 | 53 | 144 | 138 |
| Historical | 5+ | 2 | 5 | 7 | 8 |
| Historical | 6 | 0 | 0 | 0 | 0 |
| Development | Mean Max Hit | 3.130 | 3.099 | 3.781 | 3.750 |
| Development | 4+ | 45 | 45 | 139 | 142 |
| Development | 5+ | 1 | 1 | 11 | 2 |
| Development | 6 | 0 | 0 | 0 | 0 |

최종 결정은 **D. 둘 다 종료**다.

1. Combinadic Rank Dynamics는 새로운 상태 표현으로서 유효하지만 forward prediction 정보력은 확인되지 않았다.
2. Reverse Seed Basin은 exact seed 주변 밀집과 다음 center 운반 모두 실패했다.
3. 두 약한 이상점을 결합하면 검정 수와 자유도만 늘어나므로 Hybrid를 만들지 않는다.
4. Locked와 Blind는 그대로 봉인한다.

## 6. 생성 산출물

### Combinadic

- `ranks.csv`
- `predictions.csv`
- `walk_forward.csv`
- `metrics.json`
- `round-vs-rank.png`
- `round-vs-delta-rank.png`
- `predicted-vs-actual-rank.png`
- `rank-prediction-error.png`
- `rolling-rank-error.png`

### Seed Basin

- `exact_seeds.csv`
- `basin_summary.csv`
- `basin_predictions.csv`
- `walk_forward.csv`
- `metrics.json`
- basin center/error/width/density/distance PNG 6종

### Comparison

- `algorithm_comparison.csv`
- `summary.json`

## 7. 결과 무결성

| 파일 | SHA-256 |
|---|---|
| Combinadic `metrics.json` | `89c92f37cedc18881d85d3b062174c04e71edafd32d289f7540d7021b157d2ef` |
| Combinadic `walk_forward.csv` | `41db8a9ff866bdd46d2928c22700f35f7a2fbcf67c51a59b7094d3510094cb38` |
| Seed Basin `metrics.json` | `6b36317e2c10296c6de66cd6d7a69ff3df72f70f23684962259203fd6d9c622f` |
| Seed Basin `walk_forward.csv` | `5ccbc7ce7769de81a67a4d19cac0c25a90f2eea62db8c2fc1e44fc23707961f3` |
| Comparison `summary.json` | `ada3c41a6e478a8ed2eb46bb59d97680c0323a5aba41c9177e74ecfe5a34b602` |
| Comparison CSV | `ba31b35ab69e9c2e6cc24d5fe853f8d32c16831bd8f2a3aba9847d069b5e7c57` |

## 8. 재현 명령

```bash
PYTHONPATH=src python -m uriel_v2 combinadic-rank \
  --data lotto.xlsx \
  --start-round 852 --end-round 1235 \
  --minimum-history 200 \
  --split-round 1044 \
  --seed 20260814 \
  --output artifacts

PYTHONPATH=src python -m uriel_v2 seed-basin \
  --data lotto.xlsx \
  --landscape outputs/reverse-dataset/HISTORICAL/reverse-hit-seeds.csv \
  --landscape outputs/reverse-dataset/DEVELOPMENT/reverse-hit-seeds.csv \
  --start-round 852 --end-round 1235 \
  --minimum-history 32 \
  --split-round 1044 \
  --seed 20260814 \
  --output artifacts

PYTHONPATH=src python -m uriel_v2 compare-experiments \
  --combinadic-metrics artifacts/combinadic/RUN/metrics.json \
  --seed-basin-metrics artifacts/seed_basin/RUN/metrics.json \
  --output artifacts

PYTHONPATH=src python -m unittest discover -s tests -v
```

최신 `main`의 Seed Field 테스트까지 포함한 최종 테스트는 36개 모두 통과했다.

## 9. 한계

- Seed Basin은 `[0, 1,000,000)` seed 공간에서 얻은 landscape에 한정된다.
- basin 입력은 저장된 4+/5+/6-hit seed이며 0~3-hit 전체 행은 보존하지 않았다. 전체 탐색 건수는 알고 있으므로 4+/5+ density 비교는 가능하지만, full hit entropy는 계산하지 않았다.
- exact seed는 416회에서 50개로 희소해 exact-distance 검정력은 제한적이다.
- Candidate window는 실제 구매 후보가 아니라 가설의 국소 상태 공간을 동일 budget random과 비교하기 위한 진단이다.
- 다중 구성요소와 budget을 확인했기 때문에 nominal p 하나만으로 승격하지 않았다.
- 1236회 정답은 이번 1–1235회 walk-forward 실험에 사용하지 않았다.

## 10. 결론

당첨 조합을 Combinadic rank로 바꾸거나, 정답 근접 seed를 basin으로 바꿔도 Random과 구분되는 반복 가능한 forward structure는 확인되지 않았다.

이번 결과의 가치는 성공한 공식을 얻은 데 있지 않다. 앞으로 다음 항목을 다시 섞어 탐색하지 않아도 된다는 점에 있다.

- raw/delta/modulo/bit 기반 rank continuation
- exact seed 주변의 수치적 근접 basin
- weighted basin center의 단순 시계열 운반
- 두 실패 공간의 Hybrid

새 실험을 시작한다면 이 두 상태 공간의 연장선이 아니라, 데이터 표현이나 검정 질문 자체가 다른 가설이어야 한다.
