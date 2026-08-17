# Uriel v2 — Top30 Broad-Area Retrieval 독립 검증 보고서

- 작성일: 2026-08-18 (Asia/Seoul)
- 대상 저장소: `cjftya/uriel-v2`
- 구현 commit: `b80bbe59c746d6a50b8e4fd0a3c4ef3bec00bd1c`
- 동결 Motif 기준 commit: `b16675aa9635fd70df5617b9e36e5e3ce0a50dd0`
- 데이터: `lotto.xlsx`, 1–1235회
- Seen 재현: 852–1235회
- Confirmatory Locked: 660–851회
- Additional Blind: 468–659회, **미개방**
- 최종 판정: **C. NO SIGNAL**
- 후속 Top30 내부 reranker Phase: **진행 금지**

## Executive Summary

이번 실험의 질문은 하나였다.

> 동결된 Multi-scale Motif Top30이 Random보다 넓은 winning-number 영역을 fresh holdout에서도 반복적으로 포착하는가?

답은 **아니다**.

Seen 852–1235회는 기존 결과를 정확히 재현했다.

- Seen-Historical opportunity: 58회, Top30 총 237개, 평균 `4.0862 / 6`
- Seen-Development opportunity: 61회, Top30 총 269개, 평균 `4.4098 / 6`

그러나 사전등록 후 처음 개방한 Locked 660–851회에서는 효과가 사실상 Random 수준으로 축소됐다.

| 구간 | Opportunity | Coverage | Mean hit | Lift | Mean p | 5+ | 5+ p | Exact-6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pooled Locked | 64 / 192 | 33.33% | 4.0313 | `+0.0313` | 0.4319 | 24 / 64, 37.50% | 0.2913 | 2 / 64, 3.13% |

Random Top30 기준은 평균 `4.000`, 5+ `33.534%`, Exact-6 `7.290%`다. Locked의 평균 lift는 `+0.0313`에 불과했고 통계적으로 구분되지 않았다. 5+도 Random 수준이었으며, Exact-6는 오히려 Random 기대보다 낮았다.

96회 block별로도 반복성이 없었다.

- Locked-A: `4.0952`, lift `+0.0952`
- Locked-B: `4.0000`, lift `0.0000`

따라서 “두 block의 mean-hit lift가 모두 양수” 조건도 실패했다. 전체 SUCCESS 8개 조건 중 통과한 것은 표본 수와 coverage 두 개뿐이다.

계획서의 NO SIGNAL 정의인 “mean-hit test 실패와 5+가 Random 수준”, “Seen Development에서만 크고 fresh Locked에서 소멸”에 해당한다. Additional Blind 468–659는 열지 않았고, Top30 broad-area retrieval 가설을 종료한다.

## 1. 동결 설정과 실행 순서

다음 설정은 변경하지 않았다.

```text
query length = 13
historical lengths = 10, 13, 16
Top-K motifs = 40
minimum temporal separation = 100
views = Raw / Grid / Circle / Distribution / Transition / Context
confidence threshold = 0.011722291804
candidate size = 30
candidate score / tie-break = 기존 stable ranking 그대로
motif seed = 20260814
paired random seed = 20260818
paired random iterations = 100,000
bootstrap iterations = 10,000
```

실행 순서는 다음과 같이 강제했다.

1. 데이터와 동결 Motif 구현 hash 검증
2. Seen 852–1235 prediction 재생성 및 기존 집계 재현
3. `preregistration.json` 기록 및 SHA-256 고정
4. Locked 660–851 1회 평가
5. Locked SUCCESS일 때만 Blind 468–659 자동 개방
6. 통계·진단·무결성 산출물 생성

`--force-blind` 같은 우회 옵션은 구현하지 않았다.

## 2. Source validation과 Seen 재현

### 2.1 입력과 동결 구현

| 항목 | SHA-256 | 결과 |
|---|---|---|
| `lotto.xlsx` | `7efe5e232c7d4ed347b0377726686adad93ebe0581f3d03257cf4e2f83836db4` | 기준값 일치 |
| 기준 commit의 `irregular_motif.py` | `3e9463948e1f6371b2ec2ea16ba70c85711f74d9847ff88d04118fc7062da6ba` | 현재 core와 일치 |
| 현재 동결 core | `3e9463948e1f6371b2ec2ea16ba70c85711f74d9847ff88d04118fc7062da6ba` | 기준 commit과 byte 일치 |

기존 대용량 `artifacts/`는 `.gitignore` 대상이므로 fresh checkout에는 없었다. 계획서의 허용 경로에 따라 configuration selection을 다시 수행하지 않고 `multiview_long`을 직접 고정해 Seen prediction만 재생성했다.

원본 source artifact의 byte-to-byte candidate 비교는 파일 부재로 수행할 수 없었다. 대신 다음 두 조건을 모두 확인했다.

1. 기준 commit의 핵심 Motif 구현과 현재 구현이 byte 수준에서 일치
2. 공개된 Seen opportunity count와 Top30 hit total이 두 cohort 모두 정확히 일치

재생성한 Seen candidate Top30 canonical SHA-256은 다음과 같다.

```text
ea5525289e5033fb0c3cdecafaec318ce7abfb836f3d70dd9da691d3e9cea37d
```

### 2.2 Seen 재현 결과

| Cohort | Expected opportunity | Actual | Expected Top30 hits | Actual | Mean hit | 결과 |
|---|---:|---:|---:|---:|---:|---|
| Seen-Historical 852–1043 | 58 | 58 | 237 | 237 | 4.0862 | PASS |
| Seen-Development 1044–1235 | 61 | 61 | 269 | 269 | 4.4098 | PASS |

Seen 재현이 통과한 뒤에만 사전등록 파일을 생성했다.

```text
preregistration SHA-256
= 2fe99513ed6e0c8e2c68ec4284f0363d1a4b5b701bc64399c026c85eff045dd3

source_validation SHA-256
= e2a529f2a6c2bcacdfb39276b6299f8587c1eac206d9de2811698e1c70e5d9c0
```

## 3. Locked Primary 결과

### 3.1 Pooled Locked

| 지표 | Locked | Random | 차이 | 판정 |
|---|---:|---:|---:|---|
| Opportunity count | 64 | 최소 40 | +24 | 통과 |
| Opportunity coverage | 33.33% | 허용 20–45% | 범위 내 | 통과 |
| Mean hit | 4.0313 | 4.0000 | `+0.0313` | 기준 4.20 실패 |
| Mean-hit one-sided p | 0.4319 | ≤0.05 필요 | - | 실패 |
| Inclusion rate | 67.19% | 66.67% | `+0.52pp` | 효과 미미 |
| 5+ count / rate | 24 / 64, 37.50% | 33.534% | `+3.97pp` | 기준 40% 실패 |
| 5+ exact binomial p | 0.2913 | ≤0.05 필요 | - | 실패 |
| Exact-6 count / rate | 2 / 64, 3.13% | 7.290% | `-4.17pp` | guardrail 실패 |

Mean hit bootstrap 95% CI는 `[3.7813, 4.2813]`이며 Random 평균 4.0을 넓게 포함한다. 표준화 effect size는 `d=0.0305`로 사실상 0이다.

Exact-6 Clopper–Pearson 95% CI는 `[0.38%, 10.84%]`다. 표본 불확실성은 크지만 관측 rate 자체가 Random보다 낮으므로 SUCCESS guardrail을 통과하지 못한다.

### 3.2 Locked 96회 block

| Block | Opportunity | Coverage | Mean hit | Lift | 5+ rate | Exact-6 rate |
|---|---:|---:|---:|---:|---:|---:|
| Locked-A 660–755 | 21 | 21.88% | 4.0952 | `+0.0952` | 23.81% | 4.76% |
| Locked-B 756–851 | 43 | 44.79% | 4.0000 | `0.0000` | 44.19% | 2.33% |

두 block은 서로 다른 방식으로 실패했다.

- Locked-A는 mean hit이 약간 높지만 5+가 Random보다 크게 낮다.
- Locked-B는 5+가 높지만 mean hit이 정확히 Random이고 Exact-6가 더 낮다.
- 두 block 모두 같은 방향으로 mean-hit lift를 재현하지 못했다.
- Locked-B coverage `44.79%`는 허용 상한 `45%` 바로 아래다. 많은 회차를 opportunity로 선택하고도 평균 성능은 Random이었다.

## 4. Multi-block 안정성

| Block | Opportunity | Coverage | Mean hit | Lift | 5+ rate | Exact-6 rate |
|---|---:|---:|---:|---:|---:|---:|
| Seen-A 852–947 | 23 | 23.96% | 4.1304 | `+0.1304` | 39.13% | 4.35% |
| Seen-B 948–1043 | 35 | 36.46% | 4.0571 | `+0.0571` | 37.14% | 8.57% |
| Seen-C 1044–1139 | 30 | 31.25% | 4.5000 | `+0.5000` | 50.00% | 10.00% |
| Seen-D 1140–1235 | 31 | 32.29% | 4.3226 | `+0.3226` | 45.16% | 19.35% |
| Locked-A 660–755 | 21 | 21.88% | 4.0952 | `+0.0952` | 23.81% | 4.76% |
| Locked-B 756–851 | 43 | 44.79% | 4.0000 | `0.0000` | 44.19% | 2.33% |

강한 효과는 Seen-Development의 Seen-C/D에 집중돼 있었다. 시간을 거슬러 올라간 fresh Locked에서는 lift가 `+0.0313`으로 축소됐고 한 block은 정확히 0이 됐다.

이는 Top30 성능이 지속적인 broad-area retrieval 규칙이라기보다 Development 구간에 집중된 국소 burst였다는 해석과 일치한다.

## 5. Opportunity enrichment

| Cohort | Subset | n | Mean hit | Lift | 5+ rate | Exact-6 rate |
|---|---|---:|---:|---:|---:|---:|
| Seen-Historical | All | 192 | 4.0156 | `+0.0156` | 31.77% | 4.17% |
| Seen-Historical | Opportunity | 58 | 4.0862 | `+0.0862` | 37.93% | 6.90% |
| Seen-Historical | Non-opportunity | 134 | 3.9851 | `-0.0149` | 29.10% | 2.99% |
| Seen-Development | All | 192 | 4.1094 | `+0.1094` | 36.98% | 9.38% |
| Seen-Development | Opportunity | 61 | 4.4098 | `+0.4098` | 47.54% | 14.75% |
| Seen-Development | Non-opportunity | 131 | 3.9695 | `-0.0305` | 32.06% | 6.87% |
| Locked | All | 192 | 3.9948 | `-0.0052` | 34.38% | 4.17% |
| Locked | Opportunity | 64 | 4.0313 | `+0.0313` | 37.50% | 3.13% |
| Locked | Non-opportunity | 128 | 3.9766 | `-0.0234` | 32.81% | 4.69% |

Locked opportunity가 All rounds보다 평균 `+0.0365`개 높기는 하다. 그러나 opportunity 자체가 Random보다 구분되지 않고, All rounds는 Random보다 `-0.0052` 낮다. 따라서 opportunity detection이 실용적인 broad-area signal을 농축한다고 주장할 수 없다.

특히 Locked의 Exact-6는 opportunity `3.13%`가 non-opportunity `4.69%`보다도 낮다.

## 6. Candidate funnel 진단

Opportunity에 속한 모든 실제 winning number를 같은 frozen score rank에서 추적했다.

| Cohort | Top30 | Top25 | Top20 | Top15 | Top10 |
|---|---:|---:|---:|---:|---:|
| Seen-Historical | 68.10% | 55.46% | 45.69% | 34.20% | 22.13% |
| Seen-Development | 73.50% | 60.11% | 50.00% | 36.61% | 25.41% |
| Locked | 67.19% | 51.82% | 42.45% | 30.73% | 19.27% |

Locked에서는 Top30부터 Random 기대 `66.67%`와 거의 같았고, Top25 이하에서는 Seen 두 cohort보다 계속 낮았다.

이 결과는 “Top30 broad-area retrieval은 fresh holdout에서도 살아 있고 내부 ranking만 병목”이라는 진단 가설도 지지하지 않는다. fresh holdout에서는 Top30 자체가 먼저 Random 수준으로 돌아갔다.

따라서 Top30 내부 reranker나 Top20/Top15 압축 Phase를 진행하지 않는다.

## 7. Exact-6 회차

Locked opportunity 중 Top30에 여섯 번호가 모두 포함된 회차는 두 개였다.

| 회차 | 당첨번호 | Winning ranks | 최대 rank |
|---:|---|---|---:|
| 674 | 9, 10, 14, 25, 27, 31 | 20, 15, 13, 12, 28, 27 | 28 |
| 815 | 17, 21, 25, 26, 27, 36 | 21, 17, 1, 4, 3, 19 | 21 |

674회는 여섯 번호 중 두 개가 rank 27–28에 있어 Top25로 줄이면 exact inclusion이 바로 사라진다. 815회는 Top21까지 유지된다.

이 두 회차는 appendix용 설명 사례일 뿐 새 feature, weight, threshold 또는 SUCCESS 규칙에 사용하지 않았다. 관측 Exact-6 rate가 Random보다 낮으므로 사례 존재만으로 신호를 주장할 수 없다.

## 8. Surrogate 진단

| Cohort | Surrogate | Observed | Surrogate mean | Lift | p | FDR q |
|---|---|---:|---:|---:|---:|---:|
| Locked | Round shuffle | 4.0313 | 3.9511 | `+0.0802` | 0.2366 | 0.3549 |
| Locked | Block shuffle | 4.0313 | 3.9328 | `+0.0985` | 0.1746 | 0.2993 |
| Locked | Candidate score permutation | 4.0313 | 4.0013 | `+0.0300` | 0.4392 | 0.4392 |
| Locked | Feature-preserving | 4.0313 | 3.9987 | `+0.0326` | 0.4330 | 0.4392 |

Locked에서는 네 surrogate 중 어느 것도 유의하지 않았다. Development에서만 네 surrogate가 유의했던 기존 방향은 fresh holdout에서 반복되지 않았다.

SUCCESS 판정에는 surrogate 중 유리한 결과를 선택해 사용하지 않았다.

## 9. SUCCESS 조건 판정

| 조건 | 기준 | Locked | 결과 |
|---|---|---:|---|
| Opportunity count | ≥40 | 64 | PASS |
| Coverage | 20–45% | 33.33% | PASS |
| Mean hit | ≥4.20 | 4.0313 | FAIL |
| Mean-hit p | ≤0.05 | 0.4319 | FAIL |
| 5+ rate | ≥40% | 37.50% | FAIL |
| 5+ p | ≤0.05 | 0.2913 | FAIL |
| 두 block lift | 모두 >0 | `+0.0952`, `0.0000` | FAIL |
| Exact-6 guardrail | ≥7.290% | 3.13% | FAIL |

통과는 **2/8**이다.

최종 판정은 다음과 같다.

```text
C. NO SIGNAL
```

Additional Blind는 열지 않았다.

```text
Blind-A 468–563: SEALED
Blind-B 564–659: SEALED
Blind prediction rows: 0
Blind target-label access: 0
```

## 10. 데이터 누수와 무결성 검증

최종 독립 재계산은 **17/17 통과**했다.

- 평가 prediction: 576회 × 45개 번호 = 25,920행
- target key: 576개, 중복 0
- target별 rank: 정확히 1–45
- target별 Top30: 중복 없는 30개, 범위 1–45
- 모든 target에서 `history_end_round < target_round`
- prediction 생성 후에만 target label 교차
- Seen-Historical: 58 opportunity / 237 hits 재확인
- Seen-Development: 61 opportunity / 269 hits 재확인
- Locked: 64 opportunity / 총 258 hits 재확인
- Locked 5+: 24회 재확인
- Locked Exact-6: 2회 재확인
- Blind prediction·metric·label access: 0
- block 경계: Seen 4개 + Locked 2개 정확히 일치
- 예상 밖 결측: 0

테스트와 환경 검증:

```text
pytest: 71 passed
pip check: No broken requirements found
PNG: 10개, 최소 1360×928
그래프 render QA: block lift / funnel / decision chart 원본 확인
```

## 11. 판정 집계 수정 이력

첫 실행 로그는 Locked 결과를 `B. WEAK SIGNAL`로 출력했다. 독립 검산에서 `_success_decision`의 NO SIGNAL 분기가 계획서보다 좁게 구현된 것을 발견했다.

잘못된 분기는 다음 경우를 WEAK로 남겼다.

```text
mean lift가 아주 작게 양수이고
5+ lift도 아주 작게 양수이지만
두 검정은 모두 실패한 경우
```

계획서는 이 경우를 명시적으로 NO SIGNAL로 정의한다.

```text
Locked mean-hit test 실패 + 5+가 Random 수준
```

따라서 최종 판정 mapping을 계획서에 맞게 수정했다. 또한 Blind가 봉인돼 산출물이 없는 정상 상태를 integrity 실패로 표시하던 보고용 boolean도 바로잡았다.

중요한 제한:

- Locked prediction 재실행 없음
- target label 재개방 없음
- candidate score·ranking 변경 없음
- threshold·seed·통계량 변경 없음
- Blind 개방 없음
- `top30_predictions.parquet`, checkpoint, CSV 관측값 변경 없음

초기 `uriel.log`는 실행 당시 기록을 보존하므로 B 라벨이 남아 있다. 최종 `metrics.json`, `run_state.json`, 본 보고서와 구현 commit의 판정은 모두 **C. NO SIGNAL**로 수정됐다.

## 12. 산출물과 hash

실행 디렉터리:

```text
artifacts/top30_broad_retrieval/20260818-080532-top30-broad-retrieval/
```

주요 SHA-256:

| 파일 | SHA-256 |
|---|---|
| `preregistration.json` | `2fe99513ed6e0c8e2c68ec4284f0363d1a4b5b701bc64399c026c85eff045dd3` |
| `source_validation.json` | `e2a529f2a6c2bcacdfb39276b6299f8587c1eac206d9de2811698e1c70e5d9c0` |
| `metrics.json` | `1def122b32e3c9e672e35ae9e30d89ec042a91903042e963fc2e6fd43eaf9611` |
| `top30_predictions.parquet` | `5045e5526b05990406a815160ebdbd3f51f457db8f33adb3d15f4b92bea4e87b` |
| `opportunity_rounds.csv` | `e5af150bb3483e3147df7d7d5ee0d73dc4cc8f8e470bddd4c471d92cb46df6e6` |
| `block_metrics.csv` | `7efe408141ae1d413851e7d79a216f6a94b19655cd047b5e7a7d7a0c0ee361e6` |
| `candidate_funnel.csv` | `baecb74273dce56d77c65e8b9b6d37b1c2cffb5ae6cab20df915670c2c0af43e` |

전체 산출물:

```text
preregistration.json
source_validation.json
top30_predictions.parquet
opportunity_rounds.csv
block_metrics.csv
cohort_metrics.csv
hit_distribution.csv
exact6_rounds.csv
candidate_funnel.csv
winning_number_ranks.csv
random_baseline.csv
surrogate_results.csv
checkpoint.jsonl
run_state.json
target_label_access.csv
metrics.json
plots/*.png (10개)
uriel.log
```

## 13. 재현 명령

```bash
PYTHONPATH=src \
MPLCONFIGDIR=/tmp/uriel-top30-mpl \
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
  --output artifacts \
  --verbose

python -m pytest -q
python -m pip check
```

실행 환경은 Python 3.12.13이며 전체 실험 시간은 327.23초였다.

Locked/Blind는 독립 증거이므로 위 명령을 반복해 새 통계 증거로 세면 안 된다. 이 보고서 이후 660–851은 더 이상 holdout이 아니다.

## 14. 최종 결론

동결된 Multi-scale Motif Top30은 Seen-Development에서 `4.4098 / 6`까지 올라갔지만 fresh Locked에서는 `4.0313 / 6`으로 Random 수준에 돌아왔다.

Top30 내부 ranking만의 병목으로 볼 근거도 남지 않았다. Locked funnel은 Top30부터 Seen보다 낮았고, All rounds 성능도 Random 아래였다.

따라서 최종 결정은 다음과 같다.

1. **Top30 Broad-Area Retrieval: C. NO SIGNAL**
2. **Additional Blind 468–659 봉인 유지**
3. **Top30 내부 reranker / diverse Top20·Top15 / 10-ticket compression 진행 금지**
4. **Opportunity, Regime, Seed, Combinadic, Hybrid를 재도입해 이 결과를 구제하지 않음**
5. **새로운 독립 아이디어 없이는 현 Motif 계열 종료**

이번 실험에서 남은 정보는 “Development의 높은 Top30 포함률은 독립 구간에서 재현되지 않았다”는 부정적 결론이다. 높은 포함률 자체가 45개 중 30개를 선택하는 넓은 pool의 기저확률을 넘어선 안정적 예측력은 아니었다.
