# Uriel v2 — Irregular Recurrence Motif & Regime Transition 실험 보고서

- 작성일: 2026-08-14 (Asia/Seoul)
- 대상 저장소: `cjftya/uriel-v2`
- 데이터: `lotto.xlsx`, 1–1235회
- 평가 구간: Historical 852–1043회, Development 1044–1235회
- 실험 seed: `20260814`
- 최종 결정: **A. Multi-scale Motif 계속**
- 봉인 상태: **Locked 660–851 / Additional Blind 468–659 평가 target 미개방**

## Executive Summary

| 평가 축 | 판정 | 핵심 근거 |
|---|---|---|
| Multi-scale Recurrence Motif | **WEAK SIGNAL** | @20 전체 lift와 Historical 고정 opportunity lift가 두 구간에서 같은 양의 방향이지만, 전체 평균은 유의하지 않고 follow-up entropy 개선이 재현되지 않음 |
| Regime-only | **NO SIGNAL** | @20 평균 적중이 Historical과 Development 모두 random보다 낮고 opportunity filtering도 개선을 만들지 못함 |
| Regime-Switching + Motif Transition | **NO SIGNAL** | Historical @20은 random보다 낮고 Development만 소폭 높아 방향이 뒤집힘; entropy와 opportunity도 재현되지 않음 |
| Hybrid | **금지** | Motif가 SUCCESS가 아니고 Regime 축이 NO SIGNAL이므로 결합할 근거가 없음 |

질문에 대한 답은 **“지속적인 규칙은 확인되지 않았지만, 현재의 Multi-scale Motif opportunity 규칙에는 추가 검증 가치가 있는 좁은 방향성은 남았다”**다.

Motif @20 평균 적중 lift는 Historical `+0.0536`, Development `+0.0459`로 같은 방향이었다. Historical confidence 70th percentile로 고정한 opportunity subset에서도 `+0.0765`, `+0.3323`으로 방향이 같았다. Development opportunity의 평균 적중 p-value는 `0.0140`이지만 Historical은 `0.3252`이고, 네 follow-up entropy surrogate 중 두 구간에서 함께 개선된 항목은 없다. 따라서 운영 신호나 SUCCESS가 아니라 계획서 정의 그대로 **WEAK SIGNAL**이다.

Regime Transition은 Historical `-0.0981`, Development `+0.0471`로 @20 방향이 뒤집혔다. Development에서 exact-6가 5회 관측됐지만 Historical은 0회였고, 전체 평균·opportunity·entropy가 함께 개선되지 않았다. 단일 구간의 희소 사건을 신호로 승격하지 않는다.

결정 `A`는 현 설정을 운영에 사용한다는 뜻이 아니다. `multiview_long`과 opportunity threshold를 더 엄격하게 사전등록해 새로운 미래 데이터에서 한 번 더 검증할 가치만 인정한다. Locked/Blind는 열지 않고, Regime 및 Hybrid는 종료한다.

## 1. 실험 범위와 재현성

| 항목 | 값 |
|---|---|
| Python | 3.12.13 |
| Uriel v2 | 0.4.0 |
| 구현 commit (`main`) | `c92808f0a5c7dcb8a425be6768c83d9556134036` |
| Motif 실행 시간 | 734.74초 |
| Regime 최종 실행 시간 | 84.96초 |
| 데이터 SHA-256 | `7efe5e232c7d4ed347b0377726686adad93ebe0581f3d03257cf4e2f83836db4` |
| 평가 target | 384회: Historical 192 + Development 192 |
| Random / permutation / bootstrap | 각 10,000회 |
| 최종 테스트 | 46 passed |
| 검증 상태 | **Ready to share**, 22개 무결성 체크 통과 |

각 target `t`의 feature 정규화, motif retrieval, PCA, clustering, threshold 적용에는 `1..t-1`만 사용했다. Development를 본 뒤 설정을 다시 선택하거나 임계값을 바꾸지 않았다.

Locked/Blind의 의미는 **해당 구간을 이번 실험의 평가 target으로 열지 않았다**는 뜻이다. 852회 이후 target을 예측할 때 이미 과거가 된 회차는 `1..t-1` 이력에 포함된다. 이는 계획서의 `1..851 → 852` walk-forward 규칙과 일치한다.

`lotto.xlsx`는 1–1235회 1,235행, 중복 회차·결측·범위 오류가 없었다. 보너스 번호와 금액 정보는 사용하지 않았다. 1236회 정보도 사용하지 않았다.

## 2. 구현한 상태 공간과 평가 설계

### 2.1 Multi-view state

회차별 Parquet에는 330개 열을 저장했다. 회차·번호 6개와 grid mask 49개를 제외한 여섯 view는 다음과 같다.

| View | 주요 내용 |
|---|---|
| Raw | normalized number, gap, 범위·합·분산, gap entropy와 gap 위치 |
| Grid | 7×7 mask, bounding box, 중심·분산, pair 거리, 행·열 점유, symmetry·compactness |
| Circle | 절대 각도, chord·angular gap, centroid, polygon, 회전 정규화 shape |
| Distribution | 홀짝·low/high, 구간·끝수 점유, 소수·연속·same-decade, sum/spread/density |
| Transition | `t-1/t-2` overlap, number/grid/circle displacement, sum/range/dispersion/gap 변화 |
| Context | 최근 3·5·8·13회 frequency, pair concentration, grid/circle heat, entropy와 dispersion 변화 |

정확한 normalized DTW와 Derivative-DTW를 함께 계산했다. coarse resampling으로 후보를 줄인 뒤 exact distance를 계산했고, 전체 view를 너무 일찍 한 점수로 압축하지 않고 support vector를 보존했다.

### 2.2 사전등록 설정과 고정 규칙

Motif는 무차별 조합 탐색 대신 세 family만 Historical에서 비교했다.

| 설정 | Query / 과거 길이 | Top-K | 최소 separation | View | Historical score |
|---|---|---:|---:|---|---:|
| `shape_short` | 5 / 4·5·6 | 20 | 30 | Raw·Grid·Circle | -0.0607 |
| `state_medium` | 8 / 6·8·10 | 30 | 50 | Distribution·Transition·Context | +0.0224 |
| `multiview_long` | 13 / 10·13·16 | 40 | 100 | 6개 전체 | **+0.0703** |

`multiview_long`을 고정했고 Development에서는 재선택하지 않았다. opportunity threshold도 Historical confidence 70th percentile인 `0.011722291804`로 고정했다.

Regime은 KMeans/GMM의 `K=4,6,8,12,16`과 HDBSCAN 최소 cluster size 20·40을 Historical의 균등 간격 24개 calibration target에서 비교했다. 선택된 설정은 **GMM K=8**, transition query 5, 과거 길이 4·5·7, Top-30, 최소 separation 50이다. Development opportunity threshold는 Historical에서 고정한 `0.023576447426`이다.

PCA는 매 target prefix에서 결정적으로 다시 맞췄다. 장기·고공선성 prefix의 LAPACK full-SVD 실패를 재현한 뒤 deterministic randomized SVD 8차원으로 고정했고, 실패 회차와 전체 경계 prefix를 다시 검증했다.

### 2.3 Baseline과 판정

- Round shuffle, within-round random Lotto, block shuffle, feature-preserving surrogate를 사용했다.
- Candidate size 10·15·20·25·30은 같은 크기의 random subset 10,000회와 비교했다.
- Follow-up entropy는 bootstrap 95% CI, paired permutation, 네 surrogate의 Benjamini–Hochberg FDR을 기록했다.
- Historical-selected 설정과 opportunity threshold를 Development에 고정했다.
- p-value 하나, 단일 5+/6-hit, 또는 한 구간의 이상점으로 성공을 선언하지 않았다.
- `WEAK SIGNAL`은 계획서대로 전체 및 opportunity lift가 두 구간에서 같은 방향일 때만 허용했다. 이 판정은 Locked/Blind 개방 조건이 아니다.

## 3. Motif Findings — opportunity 방향은 반복됐지만 후속 상태 제약은 확인되지 않음

### 3.1 Recurrence는 보였지만 surrogate의 강한 꼬리를 넘지 못했다

Historical surrogate recurrence의 95th percentile `0.566803`을 고정 threshold로 사용했다.

| Cohort | Mean recurrence similarity | Threshold 초과 density | Mean cross-view agreement |
|---|---:|---:|---:|
| Historical | 0.543201 | 1.04% | 5.160 / 6 |
| Development | 0.543245 | 0.00% | 5.109 / 6 |

평균 cross-view agreement는 높았지만, 높은 similarity 꼬리는 Development에서 사라졌다. Historical에서도 actual density는 round shuffle `7.81%`, block shuffle `11.98%`보다 낮았다. 따라서 비슷한 구조가 보인다는 사실 자체를 신호로 해석할 수 없다.

### 3.2 Follow-up entropy는 random보다 일관되게 낮아지지 않았다

표의 Δ는 `surrogate entropy - actual entropy`다. 양수여야 motif 관측 후 후속 상태가 더 제한된다.

| Cohort | Actual entropy | Round shuffle Δ / q | Random Lotto Δ / q | Block shuffle Δ / q | Feature-preserving Δ / q |
|---|---:|---:|---:|---:|---:|
| Historical | 0.978083 | +0.000132 / 0.994 | -0.000199 / 0.994 | -0.000563 / 0.994 | -0.001094 / 0.994 |
| Development | 0.978641 | -0.000319 / 1.000 | -0.001131 / 1.000 | -0.001347 / 1.000 | -0.001706 / 1.000 |

Historical의 round shuffle 대비 감소는 `0.000132`에 불과하고 FDR q-value는 `0.994`다. Development에서는 네 항목 모두 actual entropy가 더 높다. 이번 실험의 핵심 질문인 “motif 이후 상태가 random보다 제한되는가”에는 긍정적으로 답할 수 없다.

### 3.3 Candidate Recall은 전체 평균에서 random과 구분되지 않았다

Mean hit은 여섯 당첨번호 중 candidate set에 들어온 수이며, mean Recall은 `Mean hit / 6`이다.

| Cohort | Size | Mean hit A/R | Mean Recall A/R | Lift | p | 4+ | 5+ | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical | 10 | 1.385 / 1.333 | 23.1% / 22.2% | +0.0525 | 0.230 | 2 | 1 | 0 |
| Historical | 15 | 1.995 / 2.000 | 33.2% / 33.3% | -0.0051 | 0.535 | 15 | 3 | 0 |
| Historical | 20 | 2.719 / 2.665 | 45.3% / 44.4% | +0.0536 | 0.267 | 42 | 11 | 2 |
| Historical | 25 | 3.359 / 3.334 | 56.0% / 55.6% | +0.0256 | 0.388 | 89 | 29 | 2 |
| Historical | 30 | 4.016 / 4.000 | 66.9% / 66.7% | +0.0158 | 0.436 | 144 | 61 | 8 |
| Development | 10 | 1.391 / 1.334 | 23.2% / 22.2% | +0.0568 | 0.215 | 4 | 0 | 0 |
| Development | 15 | 2.042 / 2.001 | 34.0% / 33.3% | +0.0410 | 0.313 | 15 | 2 | 0 |
| Development | 20 | 2.714 / 2.668 | 45.2% / 44.5% | +0.0459 | 0.301 | 47 | 9 | 1 |
| Development | 25 | 3.344 / 3.333 | 55.7% / 55.5% | +0.0110 | 0.459 | 85 | 31 | 6 |
| Development | 30 | 4.109 / 3.999 | 68.5% / 66.7% | +0.1102 | 0.0796 | 144 | 71 | 18 |

두 구간 @20 lift는 같은 양의 방향이지만 p-value가 `0.267`과 `0.301`이다. candidate size를 바꾸면 lift 크기와 방향도 달라진다. 전체 회차 성능은 SUCCESS 근거가 아니다.

### 3.4 고정 opportunity subset만 약한 추가 검증 가치를 남겼다

| Cohort | Opportunity rounds | Coverage | @20 mean hit A/R | Lift | p | 4+ | 5+ | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical | 58 | 30.21% | 2.741 / 2.665 | +0.0765 | 0.3252 | 15 | 4 | 1 |
| Development | 61 | 31.77% | 3.000 / 2.668 | +0.3323 | 0.0140 | 15 | 6 | 0 |

Historical에서 정한 threshold를 Development에 그대로 적용했는데 lift 방향이 유지되고 Development에서만 nominal significance가 나타났다. 이것이 `WEAK SIGNAL`의 유일한 근거다.

그러나 Historical은 유의하지 않고, entropy는 개선되지 않았으며, confidence/size/고적중 count를 여러 방식으로 확인했다. 이 결과를 운영 후보나 Locked 개방 근거로 확대하지 않는다.

### 3.5 조합 단위 진단은 별도 edge를 만들지 못했다

Top-20 번호 안의 `C(20,6)=38,760` 조합을 번호 score 합만으로 순위화했다. 조합 계층의 추가 가중치 튜닝은 하지 않았다.

| Cohort | Budget | Mean best hit | 4+ rounds | 5+ rounds | 6 rounds |
|---|---:|---:|---:|---:|---:|
| Historical | 100 | 1.792 | 5 | 0 | 0 |
| Historical | 1,000 | 2.307 | 19 | 2 | 0 |
| Historical | 10,000 | 2.667 | 40 | 5 | 1 |
| Development | 100 | 1.865 | 6 | 0 | 0 |
| Development | 1,000 | 2.359 | 15 | 1 | 0 |
| Development | 10,000 | 2.677 | 44 | 6 | 0 |

큰 budget에서 번호 후보 recall을 조합 열거로 바꾼 결과이며, 독립적인 조합 신호로 해석하지 않는다.

### 3.6 Motif 판정

**WEAK SIGNAL**

- 전체 @20 lift가 두 구간에서 작게 같은 방향이다.
- Historical에서 고정한 opportunity subset의 @20 lift도 두 구간에서 같은 방향이다.
- Development opportunity 평균 적중은 nominal `p=0.0140`이다.
- 그러나 전체 평균은 유의하지 않고 follow-up entropy 감소가 없다.
- SUCCESS, 운영 승격, Hybrid, Locked/Blind 개방 조건은 충족하지 못했다.

## 4. Regime Findings — 상태 분할과 전이 모두 재현되지 않음

### 4.1 Regime-only는 두 구간 모두 random보다 낮았다

| Cohort | @20 mean hit A/R | Lift | p | 4+ | 5+ | 6 | Opportunity lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical | 2.630 / 2.667 | -0.0372 | 0.686 | 41 | 10 | 2 | -0.0645 |
| Development | 2.625 / 2.667 | -0.0415 | 0.702 | 44 | 7 | 3 | -0.0323 |

한 시점의 regime 또는 soft membership만으로는 후보 수를 줄이지 못했다. opportunity filtering도 두 구간 모두 악화됐다.

### 4.2 Transition motif recurrence는 block structure와 구분되지 않았다

Historical pooled surrogate 95th percentile `0.905656`을 recurrence threshold로 고정했다.

| Cohort | Actual mean similarity | Actual density | Round-shuffle mean / density | Block-shuffle mean / density |
|---|---:|---:|---:|---:|
| Historical | 0.7442 | 0.00% | 0.7268 / 3.65% | 0.7909 / 15.63% |
| Development | 0.7290 | 0.00% | 0.7177 / 4.69% | 0.7892 / 19.79% |

Actual 평균은 round shuffle보다 조금 높지만 high-threshold recurrence는 0회였고 block shuffle이 더 강했다. transition family가 장기 시간 구조를 보존한 실제 데이터에서 특별하다고 볼 수 없다.

### 4.3 Transition follow-up entropy도 개선되지 않았다

| Cohort | Actual entropy | Round shuffle Δ / q | Random Lotto Δ / q | Block shuffle Δ / q | Feature-preserving Δ / q |
|---|---:|---:|---:|---:|---:|
| Historical | 0.969502 | -0.000682 / 0.845 | +0.000033 / 0.845 | +0.000129 / 0.845 | -0.000593 / 0.845 |
| Development | 0.969263 | -0.000308 / 0.973 | -0.000848 / 0.973 | -0.000459 / 0.973 | -0.001333 / 0.973 |

두 구간에서 같은 surrogate 대비 의미 있는 entropy 감소가 없다.

### 4.4 Candidate Recall은 Historical과 Development 방향이 뒤집혔다

| Cohort | Size | Mean hit A/R | Mean Recall A/R | Lift | p | 4+ | 5+ | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical | 10 | 1.385 / 1.334 | 23.1% / 22.2% | +0.0518 | 0.239 | 5 | 0 | 0 |
| Historical | 15 | 1.974 / 2.000 | 32.9% / 33.3% | -0.0257 | 0.641 | 16 | 2 | 0 |
| Historical | 20 | 2.568 / 2.666 | 42.8% / 44.4% | -0.0981 | 0.8915 | 35 | 6 | 0 |
| Historical | 25 | 3.260 / 3.334 | 54.3% / 55.6% | -0.0735 | 0.823 | 83 | 18 | 2 |
| Historical | 30 | 3.990 / 4.000 | 66.5% / 66.7% | -0.0108 | 0.563 | 135 | 55 | 15 |
| Development | 10 | 1.385 / 1.333 | 23.1% / 22.2% | +0.0521 | 0.237 | 3 | 0 | 0 |
| Development | 15 | 2.052 / 2.001 | 34.2% / 33.3% | +0.0514 | 0.269 | 20 | 5 | 0 |
| Development | 20 | 2.714 / 2.666 | 45.2% / 44.4% | +0.0471 | 0.2962 | 46 | 10 | 5 |
| Development | 25 | 3.432 / 3.333 | 57.2% / 55.5% | +0.0996 | 0.121 | 87 | 29 | 9 |
| Development | 30 | 4.120 / 4.000 | 68.7% / 66.7% | +0.1197 | 0.0691 | 138 | 72 | 17 |

Development @20 exact-6 5회는 random 기대 0.916회 대비 nominal count `p=0.0025`다. 그러나 Historical exact-6는 0회이고 @20 평균과 4+/5+는 유의하지 않다. Development opportunity subset에서도 exact-6 2회가 있었지만 평균 lift p-value는 `0.3383`이다. 다중 size·confidence·count 중 한 희소 통계를 사후 승격하지 않는다.

### 4.5 Opportunity filtering은 방향을 안정화하지 못했다

| Cohort | Opportunity rounds | Coverage | @20 mean hit A/R | Lift | p | 4+ | 5+ | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical | 58 | 30.21% | 2.379 / 2.667 | -0.2876 | 0.9762 | 10 | 1 | 0 |
| Development | 50 | 26.04% | 2.740 / 2.663 | +0.0767 | 0.3383 | 12 | 3 | 2 |

Historical에서 opportunity selection이 오히려 명확히 나빴고 Development의 작은 양의 차이도 유의하지 않다.

### 4.6 Regime 판정

- Regime-only: **NO SIGNAL**
- Regime Transition Motif: **NO SIGNAL**

GMM K=8은 Historical calibration에서 선택됐지만 full Historical 성능이 random보다 낮았다. Development의 일부 희소 적중은 Historical 방향과 재현되지 않았다. Regime count, probability, dwell, transition speed·volatility를 더 튜닝하지 않는다.

## 5. 최종 비교와 결정

Primary candidate size 20 비교다.

| Algorithm | Cohort | Mean hit lift | Opportunity lift | Entropy replicated | 판정 |
|---|---|---:|---:|---|---|
| Multi-scale Motif | Historical | +0.0536 | +0.0765 | 아니오 | WEAK SIGNAL |
| Multi-scale Motif | Development | +0.0459 | +0.3323 | 아니오 | WEAK SIGNAL |
| Regime-only | Historical | -0.0372 | -0.0645 | 아니오 | NO SIGNAL |
| Regime-only | Development | -0.0415 | -0.0323 | 아니오 | NO SIGNAL |
| Regime Transition | Historical | -0.0981 | -0.2876 | 아니오 | NO SIGNAL |
| Regime Transition | Development | +0.0471 | +0.0767 | 아니오 | NO SIGNAL |

최종 결정은 **A. Multi-scale Motif 계속**이다.

1. `multiview_long`의 intermittent opportunity 가설만 추가 validation 대상으로 남긴다.
2. 다음 검증은 query 13, 과거 길이 10·13·16, Top-40, separation 100, 6-view, confidence threshold `0.011722291804`, candidate size 20을 사전등록하고 바꾸지 않는다.
3. 다음 검증의 primary 질문은 opportunity @20 mean hit lift의 재현과 follow-up entropy 감소의 동시 성립이다.
4. Regime-only와 Regime Transition은 종료한다.
5. 두 축을 결합하는 Hybrid는 만들지 않는다.
6. Locked/Blind는 계속 봉인한다. WEAK SIGNAL은 개방 조건이 아니다.

## 6. 생성 산출물

### Motif

- `round_features.parquet`, `round_features_cache.npz`
- `config_selection.csv`, `recurrence_candidates.csv`
- `motif_predictions.csv`, `combination_diagnostics.csv`
- `walk_forward.csv`, `opportunity_rounds.csv`, `checkpoint.jsonl`
- `metrics.json`
- recurrence 4종, distance·separation·entropy·agreement·confidence·opportunity PNG 10종

### Regime

- `regime_assignments.csv`, `regime_probabilities.csv`
- `config_selection.csv`, `transition_motifs.csv`
- `regime_predictions.csv`, `walk_forward.csv`, `opportunity_rounds.csv`, `checkpoint.jsonl`
- `metrics.json`
- regime timeline·probability·transition·dwell·recurrence·confidence·opportunity PNG 7종

### Comparison / Validation

- `comparison.csv`, `summary.json`
- `validation.json`: source·target·cohort·고정 설정·threshold·metric 재계산·separation·row count·render QA 22개 통과

## 7. 결과 무결성

| 파일 | SHA-256 |
|---|---|
| `lotto.xlsx` | `7efe5e232c7d4ed347b0377726686adad93ebe0581f3d03257cf4e2f83836db4` |
| Motif `metrics.json` | `275fc18bfd132e49864767225125e90ae3a7fd3be828923d344c8a8447f976c8` |
| Motif `walk_forward.csv` | `c3f0bec7dea45020dc03cbfe43264b906af19584a430b168e35047ec3ee50464` |
| Regime `metrics.json` | `435eddcc08f2ba22a85f0e310af194503b9cb269c12517383c0e74df5f276a8b` |
| Regime `walk_forward.csv` | `9de0726e76f7a1aadcb189a59d56c9d5e7d90aa847a1b7a22121f2b0b2823d33` |
| Comparison `summary.json` | `1e1ba82f18bffc8aa7674358e66309510d83c14ef988c3d26946c2033398d912` |
| Comparison `comparison.csv` | `47560721b0a651b72f705015623bb7b01dba33b9c68e9455468e92f0af595787` |

## 8. 재현 명령

수치 라이브러리의 프로세스별 thread oversubscription을 막기 위해 BLAS thread를 1개로 고정하는 것을 권장한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/uriel-mpl \
python -m uriel_v2 irregular-motif \
  --data lotto.xlsx \
  --start-round 852 --end-round 1235 \
  --split-round 1044 \
  --workers 3 \
  --seed 20260814 \
  --output artifacts

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/uriel-mpl \
python -m uriel_v2 regime-motif \
  --data lotto.xlsx \
  --start-round 852 --end-round 1235 \
  --split-round 1044 \
  --workers 4 \
  --seed 20260814 \
  --output artifacts

python -m uriel_v2 motif-compare \
  --motif-metrics artifacts/motif/RUN/metrics.json \
  --regime-metrics artifacts/regime/RUN/metrics.json \
  --output artifacts

python -m pytest -q
```

선택 완료 뒤 중단된 walk-forward는 이전 실행 디렉터리 또는 `checkpoint.jsonl`을 `--resume-from`으로 지정해 이어갈 수 있다. 각 회차의 prediction과 match payload는 즉시 flush된다.

## 9. 한계와 해석 경계

- Motif WEAK SIGNAL은 좁은 opportunity 방향성에 한정된다. 전체 평균과 entropy는 신호를 확인하지 못했다.
- Motif 설정과 confidence threshold는 Historical에서 선택됐다. Development가 독립 검증이지만, candidate size·confidence cut·고적중 count를 함께 살폈으므로 nominal p 하나를 확정 증거로 보지 않는다.
- Regime Development의 exact-6 증가는 Historical에서 재현되지 않았다. 단일 구간의 희소 count를 모델 정보력으로 해석하지 않는다.
- Feature-preserving surrogate는 sum·홀짝·range 및 view marginal을 보존하는 근사다. 가능한 모든 생성 메커니즘을 대표하지 않는다.
- Recurrence matrix의 시각적 island나 broken diagonal은 탐색적 증거다. follow-up consistency와 candidate baseline을 통과하지 못하면 예측 신호가 아니다.
- Candidate set과 조합 budget은 가설의 정보력 진단용이며 실제 구매 성능을 뜻하지 않는다.
- Locked/Blind는 평가 target으로 사용하지 않았다. 향후 열려면 별도의 사전등록과 SUCCESS 수준 근거가 필요하다.

## 10. 결론

전반적으로 불규칙한 로또 시계열에서 멀리 떨어진 짧은 구조를 찾는 접근은, 고정 rank·seed 연속성보다 가설 형태는 적합했다. 그러나 유사 motif 이후의 상태가 random보다 안정적으로 제한된다는 핵심 증거는 아직 없다.

남은 것은 `multiview_long` opportunity subset의 작은 방향성뿐이다. 이를 **WEAK SIGNAL**로 보존하되, 현 결과로 예측 가능성을 주장하거나 설정을 더 조정하지 않는다. Regime 계열과 Hybrid는 종료하고, Locked/Blind는 봉인한 채 새로운 미래 데이터에서 동일 설정의 한 번 더 엄격한 검증만 허용한다.
