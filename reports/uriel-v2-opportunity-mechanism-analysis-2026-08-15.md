# Uriel v2 — Opportunity Mechanism Analysis 보고서

- 작성일: 2026-08-15 (Asia/Seoul)
- 대상 저장소: `cjftya/uriel-v2`
- Uriel v2: `0.5.0`
- 데이터: `lotto.xlsx`, 1–1235회
- 평가 target: Historical 852–1043회, Development 1044–1235회
- 실험 seed: `20260814`
- 구현 commit (`main`): `b16675aa9635fd70df5617b9e36e5e3ce0a50dd0`
- 최종 판정: **Opportunity Mechanism — NO SIGNAL**
- 최종 선택: **C. Opportunity 가설 종료**
- 봉인 상태: **Locked 660–851 / Additional Blind 468–659 미개방**

## Technical Summary

이번 분석의 답은 **“Stage 1 opportunity는 높은 유사도·낮은 follow-up entropy라는 정의상 구조를 가지지만, 실제 4+ 성공 opportunity를 실패 opportunity와 재현 가능하게 구분하는 Stage 2 mechanism은 확인되지 않았다”**이다.

Historical에서만 정의·선택한 Stage 2 규칙은 `candidate_top5_bottom5_gap >= 0.232077057872`였다. 이 규칙은 Historical opportunity를 58회에서 32회로 줄이며 Top20 mean-hit lift를 `+0.0761`에서 `+0.3953`으로 높였다. 그러나 Development에 임계값을 그대로 적용하자 32회가 남았고 lift는 `+0.3332`에서 `-0.0127`로 반전됐다. 4+ rate lift도 Historical `+0.1125`, Development `-0.1069`로 방향이 갈렸다.

성공(4+)과 실패(<4)를 비교한 47개 structural feature 중 평균 차이 방향이 두 구간에서 같은 항목은 18개였지만, effect-size와 FDR 기준을 함께 통과한 feature는 **0개**였다. Second-order motif도 Historical에서 성공끼리 오히려 더 멀었고(`effect=-0.6508`, permutation `p=0.9555`), Development의 약한 clustering은 유의하지 않았다(`effect=+0.3348`, `p=0.1706`).

View 진단에서는 Circle과 Transition 제거 시 opportunity lift가 두 구간에서 함께 감소했다. 그러나 이는 동결된 Top-40 match pool의 support vector 재가중 결과이며, 성공/실패 feature 재현과 Stage 2 Development 검증을 통과하지 못했다. 원인 mechanism을 확정하는 근거로 쓰지 않는다.

Top30은 별도의 잔여 현상을 보였다. Stage 1 opportunity Top30 lift는 Historical `+0.0849`, Development `+0.4105`였고, Development에서 winning number의 73.5%가 Top30 안에 있었다. 다만 primary Top20 quality filter가 실패했으므로 현재 opportunity 가설을 구제하지 않는다. 향후 Top30을 다시 연구하려면 이번 분석의 연장이 아니라 별도 사전등록 가설로 시작해야 한다.

## 1. 질문, 범위, 판정 기준

분석 질문은 계획서대로 두 개로 제한했다.

1. 왜 일부 opportunity 회차에서만 Multi-scale Motif가 random보다 강해졌는가?
2. 성공 opportunity는 실패 opportunity와 구조적으로 다른가?

Base Motif 설정은 다음과 같이 완전히 동결했다.

| 항목 | 동결 값 |
|---|---|
| Query length | 13 |
| Historical lengths | 10, 13, 16 |
| Top-K motifs | 40 |
| Minimum temporal separation | 100 |
| Views | Raw, Grid, Circle, Distribution, Transition, Context |
| Primary candidate size | 20 |
| Secondary candidate size | 30 |
| Stage 1 confidence threshold | `0.011722291804` |

Query length, Top-K, separation, confidence threshold, feature weight를 다시 탐색하지 않았다. Development 결과를 보고 규칙이나 임계값을 변경하지 않았고, Regime·Seed·Combinadic·Hybrid도 재도입하지 않았다.

Opportunity label은 Top20 실제 적중 수를 분석용으로만 사용해 `FAIL_0_2`, `HIT_3`, `HIT_4`, `HIT_5_PLUS`로 나눴다. Primary binary comparison은 `SUCCESS_4PLUS` 대 `FAIL_BELOW4`다. Target label은 feature 생성, motif family 생성, view support 재가중에 사용하지 않았다.

## 2. Opportunity는 무엇이 달랐는가

Stage 1 opportunity는 Historical 58회(30.21%), Development 61회(31.77%)였다.

| Cohort | FAIL_0_2 | HIT_3 | HIT_4 | HIT_5_PLUS | SUCCESS_4PLUS |
|---|---:|---:|---:|---:|---:|
| Historical | 22 | 21 | 11 | 4 | 15 |
| Development | 15 | 31 | 9 | 6 | 15 |

Opportunity와 non-opportunity의 차이는 두 구간에서 매우 뚜렷했다. 예를 들어 Historical의 median motif similarity 차이는 `+0.00915`, Cliff's delta `+0.512`, FDR `q=0.00043`이었고, Development도 `+0.00868`, delta `+0.431`, `q=0.00052`였다. Number entropy는 Historical `-0.00596`, delta `-0.816`, Development `-0.00685`, delta `-0.883`이었다.

그러나 이 차이는 대부분 **confidence 정의의 재표현**이다. Confidence 자체가 top similarity, `1 - follow-up entropy`, cross-view support로 구성되므로, opportunity가 더 높은 similarity와 더 낮은 entropy를 갖는 것은 독립적인 outcome mechanism 증거가 아니다.

시간상 opportunity는 완전히 고립되지도, 안정적인 주기를 만들지도 않았다.

| Cohort | Opportunity | Isolated | Max burst | Mean gap | Max rolling 10-round density |
|---|---:|---:|---:|---:|---:|
| Historical | 58 | 21 | 6 | 3.21회 | 90% |
| Development | 61 | 25 | 4 | 3.21회 | 80% |

두 구간의 평균 gap은 거의 같지만 burst 길이와 isolated count는 달랐다. 이를 고정 주기나 continuation으로 해석하지 않았다.

## 3. 성공 Opportunity와 실패 Opportunity는 재현 가능하게 구분되지 않았다

Motif retrieval, cross-view, follow-up consensus, candidate structure의 47개 수치 feature에 대해 mean·median, bootstrap 95% CI, permutation p-value, Benjamini–Hochberg FDR, Cliff's delta, Cohen's d를 계산했다. Bootstrap과 permutation은 각 10,000회다.

두 구간에서 평균 차이 방향이 같았던 비교적 큰 effect는 다음과 같다.

| Feature | Historical difference / Cliff δ / q | Development difference / Cliff δ / q | 판정 |
|---|---|---|---|
| Motif separation median | `-11.58 / -0.228 / 0.755` | `-10.67 / -0.191 / 0.768` | 방향만 일치 |
| View variance | `+0.000053 / +0.274 / 0.755` | `+0.000063 / +0.186 / 0.719` | 방향만 일치 |
| Transition rank | `+0.293 / +0.147 / 0.755` | `+0.283 / +0.207 / 0.610` | 방향만 일치 |
| Circle similarity | `+0.00152 / +0.138 / 0.880` | `+0.00750 / +0.328 / 0.492` | 방향만 일치 |
| Grid entropy | `-0.00167 / -0.104 / 0.871` | `-0.00171 / -0.101 / 0.752` | 작은 effect |

같은 방향만으로는 충분하지 않다. 계획서의 핵심은 Historical과 Development에서 재현되는 structural distinction이다. `|Cliff's delta| >= 0.147`과 양 구간 FDR `q <= 0.10`을 함께 요구했을 때 통과 feature는 **0/47개**였다.

Candidate sharpness는 특히 불안정했다. Historical 성공 opportunity에서 score gap과 score std가 더 컸지만 Development에서는 반대였다. 이 방향 반전이 Stage 2 선택 규칙의 실패로 직접 이어졌다.

Secondary `HIT_5_PLUS` 대 `FAIL_0_2`는 Historical 4대22, Development 6대15로 표본이 작다. 계획서에 따라 탐색적 결과로만 저장했고 판정에는 사용하지 않았다.

## 4. View Ablation은 일부 단서를 보였지만 mechanism을 확정하지 못했다

View diagnostic은 새로운 motif를 검색하지 않았다. 기존의 동결 Top-40 match pool에서 각 match의 6-view support vector만 target label 없이 재가중했다. 따라서 retrieval pool 변경 효과를 포함한 완전 ablation이 아니라 **frozen-pool diagnostic**이다.

### 4.1 View 제거

| Variant | Historical opportunity n / lift | Development opportunity n / lift | ALL 대비 두 구간 동시 감소 |
|---|---:|---:|---|
| ALL | 58 / `+0.0761` | 61 / `+0.3332` | 기준 |
| ALL - Raw | 50 / `-0.0261` | 53 / `+0.3705` | 아니오 |
| ALL - Grid | 75 / `+0.0939` | 74 / `+0.1824` | 아니오 |
| ALL - Circle | 50 / `-0.0235` | 56 / `+0.3148` | **예** |
| ALL - Distribution | 85 / `+0.0963` | 79 / `+0.1447` | 아니오 |
| ALL - Transition | 61 / `+0.0542` | 63 / `+0.2716` | **예** |
| ALL - Context | 47 / `-0.0084` | 50 / `+0.3531` | 아니오 |

Circle과 Transition을 제거하면 lift가 양 구간에서 낮아졌다. 반면 Raw와 Context 제거는 Historical만 음수가 됐고 Development는 유지 또는 증가했다. View별 contribution이 cohort에 따라 달라 mechanism이 안정적이지 않다.

### 4.2 Single-view와 제한 pair

- `Transition-only` lift는 Historical `+0.1158`, Development `+0.2709`였지만 opportunity coverage가 35.9%와 40.1%로 달랐고, 성공/실패 structural feature로 재현되지 않았다.
- `Grid-only`는 Historical `+0.2738`, Development `+0.3777`이었지만 Development opportunity가 24회로 SUCCESS 최소 표본 30회에 못 미쳤다.
- `Distribution-only`는 두 구간 모두 음수였다(`-0.0840`, `-0.0598`).
- 제한 pair 중 `Circle + Transition`은 양 구간 positive lift(`+0.1188`, `+0.1574`)였으나 p-value `0.1863`, `0.1171`로 구분되지 않았다.
- `Grid + Transition`은 Historical `-0.1103`, Development `+0.2570`으로 방향이 뒤집혔다.

따라서 Circle·Transition은 후보 생성에 관여하는 단서일 수 있지만, 성공 opportunity만 골라내는 안정적인 mechanism으로 승격할 수 없다.

## 5. Motif Family와 Second-order Motif는 성공 구조를 만들지 못했다

Motif family는 target 결과를 보지 않고 similarity support vector만으로 `shape-dominant`, `transition-dominant`, `context-dominant`, `balanced-multiview`, `high-agreement`, `low-agreement`, `high-similarity-low-support`, `moderate-similarity-wide-support`로 분류했다.

실제 match 대부분은 `moderate-similarity-wide-support`였다(Historical 5,667건, Development 5,619건). 이 family는 모든 Stage 1 opportunity에 나타났고 회차 단위 random lift는 Historical `+0.0521`, Development `+0.0476`에 불과했다. `transition-dominant`는 두 구간 모두 음수(`-0.2222`, `-0.2167`)였다. 드문 family의 높은 lift는 round 수가 너무 작거나 전 회차 outcome과 혼재해 mechanism 근거가 아니다.

Second-order motif는 각 opportunity를 similarity, 6-view support, separation, entropy vector, candidate score distribution, motif family distribution으로 표현했다. Historical 평균·표준편차로 표준화하고 Development에 그대로 적용했다.

| Cohort | Opportunity / success | Success–Failure minus Success–Success distance | Label permutation p | 해석 |
|---|---:|---:|---:|---|
| Historical | 58 / 15 | `-0.6508` | `0.9555` | 성공끼리 더 멂 |
| Development | 61 / 15 | `+0.3348` | `0.1706` | 약한 clustering, 유의하지 않음 |

두 구간의 방향이 다르고 어느 쪽도 label permutation과 구분되지 않았다. 성공 opportunity 자체가 반복되는 2차 구조를 가진다는 가설은 지지되지 않았다.

## 6. Stage 2 Quality Filter는 Development에서 실패했다

최대 5개 후보 규칙만 만들었다. Feature 방향과 임계값은 Historical opportunity에서만 정했고, Development는 선택에 사용하지 않았다.

| Rule | Historical n / @20 lift | Development n / @20 lift | 선택 |
|---|---:|---:|---|
| R1 Agreement | 33 / `+0.0938` | 38 / `+0.3853` | 아니오 |
| R2 Consensus | 32 / `+0.0541` | 35 / `+0.2740` | 아니오 |
| R3 Sharpness | 32 / `+0.3939` | 32 / `-0.0124` | **Historical 최고로 선택** |
| R4 Agreement + Consensus | 40 / `+0.0118` | 39 / `+0.3058` | 아니오 |
| R5 Recurrence + Sharpness | 33 / `+0.0299` | 39 / `+0.1805` | 아니오 |

Development 결과를 본 뒤 R1으로 바꾸면 양의 결과를 고르는 사후 선택이 된다. 사전 규칙에 따라 Historical @20 lift가 가장 큰 R3을 고정했다.

R3의 조건은 다음 하나다.

```text
candidate_top5_bottom5_gap >= 0.23207705787211091
```

### 6.1 Primary Top20

| Cohort | Stage | n | Mean hit | Random mean | Lift | p | 4+ rate | 4+ rate lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Historical | Stage 1 | 58 | 2.741 | 2.665 | `+0.0761` | 0.3279 | 25.86% | `+2.75pp` |
| Historical | Stage 2 | 32 | 3.063 | 2.667 | `+0.3953` | 0.0276 | 34.38% | `+11.25pp` |
| Development | Stage 1 | 61 | 3.000 | 2.667 | `+0.3332` | 0.0153 | 24.59% | `+1.52pp` |
| Development | Stage 2 | 32 | 2.656 | 2.669 | `-0.0127` | 0.5535 | 12.50% | `-10.69pp` |

Development 최소 표본 30회는 충족했지만, lift와 4+ rate가 모두 반전됐다. Historical에서 candidate score가 더 뾰족한 회차를 고른 규칙은 Development에서 오히려 high-score false positive를 강화했다.

### 6.2 Secondary Top30

| Cohort | Stage | n | Mean hit | Lift | p |
|---|---|---:|---:|---:|---:|
| Historical | Stage 1 | 58 | 4.086 | `+0.0849` | 0.2972 |
| Historical | Stage 2 | 32 | 4.344 | `+0.3439` | 0.0375 |
| Development | Stage 1 | 61 | 4.410 | `+0.4105` | 0.0013 |
| Development | Stage 2 | 32 | 4.219 | `+0.2175` | 0.1439 |

Top30에서는 양의 방향이 남지만 R3가 Development 성능을 개선하지 않는다. Motif가 좁은 Top20 순위보다 넓은 후보 영역을 잡는 데 상대적으로 강할 수 있다는 secondary 관찰은 남지만, primary Stage 2 실패를 뒤집지 않는다.

## 7. Candidate Funnel, Missing Winner, False Positive

119개 opportunity의 실제 당첨번호 714개를 Top30→25→20→15→10 funnel로 추적했다.

| Cohort | Top30 | Top25 | Top20 | Top15 | Top10 |
|---|---:|---:|---:|---:|---:|
| Historical | 68.1% | 55.5% | 45.7% | 34.2% | 22.1% |
| Development | 73.5% | 60.1% | 50.0% | 36.6% | 25.4% |

Top20 밖 missing winner는 372건이었고 평균 rank는 32.80이었다. Historical 189건의 평균 rank는 33.08, Development 183건은 32.50이었다. 가장 강한 support view는 Context가 Historical 159건, Development 134건으로 압도적이었다.

Top20 안 high-score false positive는 2,038건이었다. Strongest view가 Context인 경우가 Historical 879/1,001건, Development 783/1,037건이었다. Dominant motif family도 `moderate-similarity-wide-support`가 95% 이상이었다.

이 결과는 Context와 wide-support motif가 넓은 유망 영역을 만드는 동시에 많은 false positive를 함께 올리는 구조와 일치한다. 다만 Context 제거 ablation은 Historical만 악화시키고 Development는 유지했으므로, 단순히 Context를 제거하는 처방은 재현되지 않는다.

## 8. SUCCESS 조건과 최종 판정

| SUCCESS 조건 | 결과 |
|---|---|
| Historical에서 규칙 정의, Development 고정 적용 | 통과 |
| Development Stage 2 opportunity ≥30 | 통과: 32회 |
| Stage 2 @20 lift > Stage 1 lift | **실패: Development에서 하락** |
| Historical / Development Stage 2 lift 같은 양의 방향 | **실패** |
| 4+ rate lift 두 구간 같은 방향 | **실패** |
| Structural feature family 재현 | **실패: 0/47** |
| View ablation mechanism 지지 | 부분 통과: Circle·Transition 제거 시 양 구간 감소 |

최종 판정은 **Opportunity Mechanism — NO SIGNAL**이다.

최종 선택은 **C. Opportunity 가설 종료**다.

R1 같은 비선택 규칙이 Development에서 좋은 결과를 보였다는 이유로 교체하지 않는다. 후보 규칙을 Development 성능으로 다시 선택하면 이번 Phase의 핵심 통제인 고정 적용을 위반한다. 현재 구조의 Stage 2 quality filter를 추가 튜닝하거나 조건을 조합하지 않는다.

Locked 660–851과 Additional Blind 468–659는 열지 않는다. 이번 평가 target은 852–1235회뿐이며, 1236회 정답도 사용하지 않았다.

## 9. 구현과 산출물

새 CLI:

```bash
python -m uriel_v2 opportunity-analysis \
  --data lotto.xlsx \
  --motif-run artifacts/motif/RUN \
  --start-round 852 --end-round 1235 \
  --split-round 1044 \
  --seed 20260814 \
  --workers 4 \
  --output artifacts
```

주요 구현 내용:

- 동결 Motif 실행의 설정·threshold·회차·cohort·hash 검증
- 47개 retrieval/cross-view/consensus/candidate feature 생성
- Opportunity/non-opportunity와 success/failure bootstrap·permutation·FDR 비교
- Frozen match-pool view ablation 7개, single-view 6개, 제한 pair 5개
- Target-independent motif family와 second-order opportunity distance
- 최대 5개 Stage 2 rule, Historical-only 선택, Development 고정 적용
- Top30 보조 지표, candidate funnel, missing winner, false positive
- 10개 PNG와 Parquet/CSV/JSON 산출물

최종 실행 디렉터리:

```text
artifacts/opportunity_analysis/20260815-124819-opportunity-analysis/
```

산출물은 계획서에 지정된 파일을 모두 생성했다.

- `opportunity_features.parquet`, `opportunity_labels.csv`
- `opportunity_non_opportunity_comparison.csv`, `success_failure_comparison.csv`
- `view_ablation.csv`, `single_view_diagnostics.csv`, `pair_interactions.csv`
- `motif_family_analysis.csv`, `second_order_motifs.csv`
- `quality_rules.csv`, `stage2_predictions.csv`
- `candidate_funnel.csv`, `missing_winners.csv`, `false_positives.csv`
- `metrics.json`, `plots/*.png`

그래프는 계획서의 10개 항목을 모두 생성했다: success/failure effect, view ablation, agreement, entropy, sharpness, second-order map, coverage/lift, timeline, funnel, missing-winner rank.

## 10. 검증과 결과 무결성

검증 상태는 **Ready to share**다.

- 최종 테스트: **51 passed**
- 의존성 검사: `pip check` 통과
- 독립 계산·산출물 검증: **38/38 통과**
- 평가 feature grain: 384회, 회차 key 384개, 중복 없음
- Historical/Development: 각 192회
- Base run opportunity/hit와 새 feature의 값: 완전 일치
- Locked/Blind target: 0건
- PNG: 10개, 최소 1536×928, contact sheet와 핵심 차트 원본 시각 점검 통과
- 같은 code commit 재실행: CSV·Parquet·PNG 전부 SHA-256 동일

예상 결측만 있었다. `opportunity_gap`은 non-opportunity에서 비어 있고, delta 4종은 각 cohort 첫 회차에서 비어 있다. 분석 feature나 label의 예상 밖 결측은 없었다.

| 파일 | SHA-256 |
|---|---|
| `lotto.xlsx` | `7efe5e232c7d4ed347b0377726686adad93ebe0581f3d03257cf4e2f83836db4` |
| Source Motif `metrics.json` | `275fc18bfd132e49864767225125e90ae3a7fd3be828923d344c8a8447f976c8` |
| Source Motif `walk_forward.csv` | `c3f0bec7dea45020dc03cbfe43264b906af19584a430b168e35047ec3ee50464` |
| Opportunity `metrics.json` | `3ac02aecb2dd2b2ccffb4906118924d70c9c1436304b7380728b43596b077013` |
| `opportunity_features.parquet` | `f96276b021657a4120560f9dedacc5451b190f33fd14527cbccb5e6cad006984` |
| `success_failure_comparison.csv` | `d12d16f743dc5190974c66c0196c32f06f72be03b366abb51a7aea1560e7b402` |
| `view_ablation.csv` | `b8166b6c9eb51cd9b8b03d62f7341ed0d19a9088df9cef097644f438c8ef4633` |
| `quality_rules.csv` | `de006c7a701cc0c394813a4bcab764a5a7eebc7635434b73b1a2922161eaaacb` |
| `stage2_predictions.csv` | `19df750aa3b89b506d639856fdfaebc9428988857152f861358de67bffbccc06` |
| `second_order_motifs.csv` | `c95ce38d2947fb72887f6b7c30673a8341e0d4571d7c3d60a0c2c92f3adb5160` |

## 11. 재현 명령

```bash
MPLCONFIGDIR=/tmp/uriel-opportunity-mpl \
python -m uriel_v2 opportunity-analysis \
  --data lotto.xlsx \
  --motif-run artifacts/motif/20260814-174919-irregular-motif \
  --start-round 852 \
  --end-round 1235 \
  --split-round 1044 \
  --seed 20260814 \
  --workers 4 \
  --output artifacts \
  --verbose

python -m pytest -q
python -m pip check
```

실행 시간은 Python 3.12.13에서 84.65초였다. `--workers`는 계획서와 CLI 호환을 위해 유지하지만, frozen-match diagnostic 자체는 결정적 단일 프로세스로 실행한다.

## 12. 한계와 불확실성

- View ablation은 frozen Top-40 match pool 재가중이다. View 제거 후 alternative motif가 Top-40에 새로 들어오는 retrieval-level 변화를 측정하지 않는다.
- Opportunity/non-opportunity 차이의 상당 부분은 confidence 정의에 내장되어 있어 outcome mechanism으로 해석할 수 없다.
- Success opportunity는 cohort별 15회뿐이다. FDR과 effect-size 재현 조건을 통과하지 못한 nominal 차이는 과대 해석하지 않는다.
- HIT_5_PLUS는 Historical 4회, Development 6회라 secondary 탐색 결과다.
- Stage 2 후보 규칙 5개를 Historical에서 비교했으므로 Historical 선택 성능에는 winner's curse가 포함될 수 있다. Development 실패가 이를 확인한다.
- Top30 enrichment는 candidate set이 45개 중 30개로 넓다. mean hit가 높아도 실제 운영 예측력이나 구매 수익을 뜻하지 않는다.
- Label permutation은 second-order clustering의 exchangeability null을 검정한다. 가능한 모든 비정상 시계열 생성 과정을 대표하지 않는다.
- 이 분석은 연관 구조를 설명하는 diagnostic이며 인과적 predictability를 확립하지 않는다.

## 13. 권고 조치

1. 현재 Stage 2 opportunity quality filter 연구를 종료한다.
2. R3 threshold, candidate sharpness 조합, 다른 percentile을 추가 튜닝하지 않는다.
3. R1을 Development 결과 때문에 사후 승격하지 않는다.
4. Locked/Blind를 계속 봉인한다.
5. 기존 `multiview_long`을 운영 예측기로 승격하지 않는다.
6. Top30 broad-area 현상을 다시 검증하려면 이번 가설과 분리한 새 계획에서 metric·threshold·최소 표본을 사전등록한다.

## 14. 후속 검토 질문

현재 계획 안에서 추가로 답해야 할 질문은 없다. 핵심 질문인 성공/실패 structural distinction은 재현되지 않았다.

남은 Top30 현상은 이번 Stage 2 가설을 계속 튜닝할 이유가 아니라, 완전히 별개의 broad-area retrieval 가설을 만들 가치가 있는지 결정할 때만 다시 검토한다. 그 경우에도 새 미래 데이터나 별도 승인된 holdout 없이 현재 Development를 다시 선택에 사용하지 않는다.

## 15. 결론

일부 회차에서 Multi-scale Motif가 random보다 강해졌던 직접적인 이유를 재현 가능한 structural mechanism으로 설명하지 못했다. Stage 1 opportunity는 더 높은 recurrence similarity와 더 낮은 entropy를 갖지만 이는 confidence 정의의 결과다. 실제 4+ 성공은 47개 feature, view ablation, motif family, second-order distance, Stage 2 filter 중 어느 경로에서도 Historical과 Development 양쪽에서 안정적으로 분리되지 않았다.

Historical에서 가장 강했던 candidate sharpness rule은 Development에서 Top20 lift와 4+ rate를 모두 악화시켰다. 따라서 **Opportunity Mechanism은 NO SIGNAL**, 최종 선택은 **C. Opportunity 가설 종료**다. Locked/Blind는 열지 않는다.
