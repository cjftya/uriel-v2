# Uriel v2 — 확률적 알고리즘 예측 모델 Phase 16 최종 실용성 평가 보고서

- 작성일: 2026-08-21 (Asia/Seoul)
- 대상 저장소: `cjftya/uriel-v2`
- Phase 16 구현 commit (`main`): `b7b667bc9b7c1e0df362caf5730b2596ffc7cc7e`
- 평가 schema: `phase16-v1`
- 평가 모델: `preregistered_multi_metric_final_viability_assessment`
- 실행 상태: **PHASE_16_PASS**
- 최종 판정: **B — PARTIAL_SUCCESS_RESEARCH_ONLY**
- 배포 가능: **아니오**
- 연구 가치: **있음**

## Executive Summary

Uriel v2의 16개 Phase를 통해 다음 조건부 분포를 예측하는 연구 파이프라인을 구축하고 최종 평가했다.

```text
p(
  Quality,
  Runtime,
  Failure
  |
  Problem Structure,
  Random Mechanism,
  Budget,
  Early Trajectory
)
```

최종 결론은 다음과 같다.

> **품질·실행시간·first-passage에 대한 예측 분포는 연구용 진단 모델로 유효하지만, 그 분포를 이용한 자동 알고리즘 선택기는 배포 가능한 수준으로 검증되지 않았다.**

10개 최종 기준 중 3개가 통과했고, 5개가 실패했으며, 1개는 관측 자료가 없어 평가 불가, 1개는 도메인 범위가 미완성이었다.

| 상태 | 기준 수 | 기준 |
|---|---:|---|
| PASS | 3 | Point prediction, marginal calibration, survival prediction |
| FAIL | 5 | Joint probability, random 대비 선택 가치, global-best 대비 선택 가치, regret, robustness |
| UNAVAILABLE | 1 | Failure risk estimability |
| INCOMPLETE | 1 | Domain expert coverage |

Point prediction에서는 두 held-out split의 품질·실행시간 모두 training-mean reference를 이겼다. Marginal calibration MAE는 전부 `0.10` 이내였고, survival C-index는 `0.833455–0.905955`, integrated Brier score는 `0.047441–0.048532`였다.

반면 joint probability는 6개 지표 중 1개만 통과했다. 자동 선택 정책의 utility gain 95% CI 하한은 random과 training-global-best 양쪽 모두에서 두 split 전부 음수였다. Training-global-best 대비 oracle regret도 두 split에서 더 컸고, cross-split 선택 일치율은 `0.561111`로 기준 `0.75`를 넘지 못했다.

따라서 현재 결과는 **예측 모델의 부분 성공**이지 **자동 선택 시스템의 성공**이 아니다. 예측기는 연구·진단 도구로 유지할 수 있지만, 선택기는 재설계 전까지 운영 의사결정에 사용하지 않는다.

## 1. 평가 범위와 해석 경계

이번 판정은 Phase 4부터 Phase 16까지 연결한 end-to-end 통합 검증 실행을 기준으로 한다.

| 항목 | 값 |
|---|---:|
| 문제 instance | 60 |
| 알고리즘 실행 | 240 |
| paired comparison | 120 |
| model feature / target 행 | 각 720 |
| 평가 split | `instance_holdout`, `family_holdout` |
| Phase 7 OOF point prediction 행 | 12,960 |
| Phase 13 joint prediction 행 | 1,440 |
| Phase 14 selection 행 | 1,440 |
| Phase 15 utility scenario | 44 |
| Phase 15 sensitivity selection 행 | 15,840 |
| Phase 16 최종 metric 행 | 31 |

이 실행은 leakage, cross-fitting, split, manifest, resume, tamper detection을 포함한 전체 파이프라인의 동작과 현재 데이터에서의 최종 판정을 함께 검증한다.

다만 외부 대규모 benchmark는 아니다. 현재 수치는 구현 통합 검증에 사용한 synthetic benchmark 범위에 대한 결과이며, 실제 배포 가능성을 주장하려면 별도의 동결된 외부 benchmark가 필요하다.

## 2. `PHASE_16_PASS`의 정확한 의미

`PHASE_16_PASS`는 다음을 뜻한다.

- Phase 6–15 입력 hash chain이 유효하다.
- 입력 산출물이 평가 도중 변경되지 않았다.
- 사전 고정된 10개 기준과 threshold가 정확히 적용됐다.
- 31개 metric이 누락·중복·비정상 수치 없이 계산됐다.
- 최종 verdict를 criterion 상태에서 다시 계산했을 때 정확히 일치했다.
- 결과물 manifest, resume, tamper detection이 정상 동작했다.

이는 모델이 배포 기준을 통과했다는 뜻이 아니다. 모델의 실질적 성공 여부는 다음 필드로 판단한다.

```json
{
  "code": "B",
  "verdict": "PARTIAL_SUCCESS_RESEARCH_ONLY",
  "deployment_ready": false,
  "research_value": true
}
```

## 3. 최종 판정 기준

최종 기준은 Phase 16 실행 전에 코드와 configuration에 고정했다. 현재 holdout 성능을 본 뒤 threshold, utility profile, 모델 또는 split을 변경하지 않았다.

| 기준 | Threshold |
|---|---:|
| Point MAE skill | `> 0.00` |
| Marginal calibration MAE | `<= 0.10` |
| Survival C-index | `>= 0.65` |
| Survival integrated Brier | `<= 0.10` |
| Joint calibration absolute error | `<= 0.10` |
| Joint NLL delta vs reference | `<= 0.00` |
| Selection utility gain CI lower | `> 0.00` |
| Selection regret delta | `<= 0.00` |
| Utility scenario retention | `>= 0.90` |
| Cross-split selection agreement | `>= 0.75` |
| Unavailable expert slots | `0` |

배포 판정인 `DEPLOYABLE_SUCCESS`는 10개 기준이 모두 `PASS`일 때만 허용한다.

## 4. 기준별 최종 결과

| 순서 | 기준 | Metric 통과 | 상태 | 배포 blocker |
|---:|---|---:|---|---|
| 1 | Point prediction skill | 4/4 | PASS | 아니오 |
| 2 | Marginal distribution calibration | 4/4 | PASS | 아니오 |
| 3 | Survival prediction | 4/4 | PASS | 아니오 |
| 4 | Joint probability generalization | 1/6 | FAIL | 예 |
| 5 | Failure risk estimability | 0/2 | UNAVAILABLE | 예 |
| 6 | Domain expert coverage | 0/1 | INCOMPLETE | 예 |
| 7 | Selection value vs random | 0/2 | FAIL | 예 |
| 8 | Selection value vs training-global-best | 0/2 | FAIL | 예 |
| 9 | Selection regret vs baselines | 2/4 | FAIL | 예 |
| 10 | Selection policy robustness | 1/2 | FAIL | 예 |

## 5. 통과한 예측 영역

### 5.1 Point prediction skill

두 held-out split에서 가장 좋은 모델은 모두 Random Forest였다. 품질과 실행시간의 MAE skill이 전부 `0`보다 커 training-mean reference를 이겼다.

| Split | Target | Best model | MAE skill | 기준 | 결과 |
|---|---|---|---:|---:|---|
| family holdout | quality | Random Forest | `0.805013` | `> 0.00` | PASS |
| family holdout | runtime | Random Forest | `0.328590` | `> 0.00` | PASS |
| instance holdout | quality | Random Forest | `0.836552` | `> 0.00` | PASS |
| instance holdout | runtime | Random Forest | `0.735538` | `> 0.00` | PASS |

품질 예측 signal은 강했고, 실행시간도 양의 일반화 signal이 확인됐다. 특히 family holdout의 runtime skill이 상대적으로 낮으므로 새로운 알고리즘 family로의 확장에서는 실행시간 예측을 계속 주의해야 한다.

### 5.2 Marginal distribution calibration

| Split | Metric | 값 | 기준 | 결과 |
|---|---|---:|---:|---|
| family holdout | Quality calibration MAE | `0.035516` | `<= 0.10` | PASS |
| family holdout | Runtime calibration MAE | `0.039881` | `<= 0.10` | PASS |
| instance holdout | Quality calibration MAE | `0.011905` | `<= 0.10` | PASS |
| instance holdout | Runtime calibration MAE | `0.006548` | `<= 0.10` | PASS |

품질과 실행시간을 각각 보는 주변분포 수준에서는 네 지표가 모두 기준을 통과했다. 따라서 단일 target의 uncertainty 진단에는 현재 모델을 연구용으로 사용할 근거가 있다.

### 5.3 First-passage survival prediction

| Split | C-index | Integrated Brier | 결과 |
|---|---:|---:|---|
| family holdout | `0.905955` | `0.047441` | PASS |
| instance holdout | `0.833455` | `0.048532` | PASS |

C-index 기준은 `>= 0.65`, integrated Brier 기준은 `<= 0.10`이다. 두 split에서 판별력과 보정이 모두 기준을 통과했다. 주어진 예산 안에서 목표 품질에 도달할 가능성과 first-passage 시점을 예측하는 부분은 현재 파이프라인의 가장 분명한 성과 중 하나다.

## 6. 실패 또는 미완성 영역

### 6.1 Joint probability generalization

| Split | Metric | 값 | 기준 | 결과 |
|---|---|---:|---:|---|
| family holdout | Absolute joint Q75 calibration error | `0.103433` | `<= 0.10` | FAIL |
| family holdout | Absolute joint Q90 calibration error | `0.126879` | `<= 0.10` | FAIL |
| family holdout | Joint NLL delta vs reference | `+0.373822` | `<= 0.00` | FAIL |
| instance holdout | Absolute joint Q75 calibration error | `0.054765` | `<= 0.10` | PASS |
| instance holdout | Absolute joint Q90 calibration error | `0.100044` | `<= 0.10` | FAIL |
| instance holdout | Joint NLL delta vs reference | `+0.094916` | `<= 0.00` | FAIL |

Marginal distribution은 통과했지만, `높은 품질 + 제한 시간 내 도달 + 무실패`를 함께 계산하는 결합확률은 일반화되지 않았다. NLL delta가 두 split 모두 양수이므로 reference보다 나빴다. Family holdout에서 오차가 더 커져 새로운 family에 대한 dependency 구조 추정이 특히 약하다.

### 6.2 Failure risk estimability

| 항목 | 결과 |
|---|---|
| 관측 failure 수 | `0` |
| Failure probability estimable | 아니오 |
| Failure type estimable | 아니오 |
| 상태 | `UNAVAILABLE — NO_OBSERVED_FAILURES` |

실패가 없었다는 사실은 실행 안정성에는 긍정적이지만, 실패 확률 모델의 성능을 증명하지는 못한다. 양성 사례가 하나도 없어 failure probability와 failure type을 경험적으로 평가할 수 없었다.

### 6.3 Domain expert coverage

| Expert slot | 상태 |
|---|---|
| sampling | 실행 가능 |
| optimization | 실행 가능 |
| matrix | 미실행 |
| natural process | 미실행 |
| stream | 미실행 |

사전 등록된 expert 중 `matrix`, `natural_process`, `stream` 세 영역이 unavailable 상태다. 현재 Mixture-of-Experts가 전체 목표 domain을 포괄한다고 볼 수 없다.

## 7. 자동 알고리즘 선택 평가

### 7.1 Random 대비 선택 가치

Primary utility profile은 `balanced`다. 성공 조건은 선택 정책의 평균 utility gain에 대한 95% CI 하한이 `0`보다 큰 것이다.

| Split | Selected vs random gain CI lower | 결과 |
|---|---:|---|
| family holdout | `-0.040427` | FAIL |
| instance holdout | `-0.020686` | FAIL |

두 split 모두 신뢰구간 하한이 음수다. 현재 선택 정책이 random보다 안정적으로 낫다고 주장할 수 없다.

### 7.2 Training-global-best 대비 선택 가치

| Split | Selected vs training-global-best gain CI lower | 결과 |
|---|---:|---|
| family holdout | `-0.203971` | FAIL |
| instance holdout | `-0.164514` | FAIL |

단순히 training에서 전역적으로 가장 좋았던 알고리즘을 계속 선택하는 baseline보다도 우월성이 입증되지 않았다.

### 7.3 Oracle regret

Regret delta가 `0` 이하여야 선택 정책의 regret이 baseline보다 나쁘지 않다.

| Split | Baseline | Regret delta | 결과 |
|---|---|---:|---|
| family holdout | random | `-0.022753` | PASS |
| family holdout | training-global-best | `+0.119722` | FAIL |
| instance holdout | random | `-0.049916` | PASS |
| instance holdout | training-global-best | `+0.093825` | FAIL |

Random 대비 regret은 줄었지만 training-global-best 대비 regret은 두 split 모두 증가했다. 복잡한 선택 정책이 단순 baseline을 이기지 못했다.

### 7.4 선택 정책 강건성

| Metric | 값 | 기준 | 결과 |
|---|---:|---:|---|
| Mean utility-scenario retention | `0.989899` | `>= 0.90` | PASS |
| Cross-split agreement rate | `0.561111` | `>= 0.75` | FAIL |

Utility 가중치 변화에는 선택이 대체로 유지됐지만, instance holdout과 family holdout 사이의 선택 일치율은 부족했다. 즉 같은 정책이 split 변화에 따라 다른 선택을 하므로 운영 환경 변화에 대한 신뢰성이 충분하지 않다.

## 8. 종합 해석

### 8.1 확인된 것

1. 문제 구조와 early trajectory에는 최종 품질과 실행시간을 예측하는 signal이 있다.
2. 품질·실행시간의 개별 확률분포는 현재 범위에서 보정 가능하다.
3. First-passage survival은 두 holdout에서 판별력과 보정 기준을 통과했다.
4. Phase 6–16의 leakage-safe, cross-fitted, artifact-hashed 평가 체인은 재현 가능하게 동작한다.

### 8.2 확인되지 않은 것

1. 품질·시간·실패를 함께 묶은 joint probability가 reference보다 낫다는 근거가 없다.
2. Failure risk를 추정할 관측 failure가 없다.
3. 모든 목표 domain을 포괄하는 expert 체계가 완성되지 않았다.
4. 자동 선택기가 random 또는 training-global-best보다 안정적으로 높은 utility를 낸다는 근거가 없다.
5. 선택 정책이 새로운 family와 instance에서 일관된 선택을 한다는 근거가 없다.

### 8.3 핵심 판단

예측 문제와 선택 문제를 분리해야 한다.

```text
예측 모델: 부분 성공
자동 선택 정책: 실패
전체 프로젝트: 연구 가치가 있는 부분 성공
운영 배포: 보류
```

좋은 예측 metric 일부가 곧바로 좋은 선택으로 이어지지는 않았다. 선택은 여러 예측값과 utility 가중치의 작은 오차가 누적되는 downstream 문제다. 현재 결과에서는 marginal prediction의 장점이 joint probability와 algorithm ranking 단계에서 보존되지 않았다.

## 9. 최종 권고

1. 현재 자동 알고리즘 선택기를 운영 환경에 배포하지 않는다.
2. Point, marginal distribution, survival predictor는 연구용 진단 모델로 유지한다.
3. Controlled failure와 timeout 사례를 의도적으로 수집한 뒤 risk model을 다시 평가한다.
4. `matrix`, `stream`, `natural_process` benchmark와 expert를 실행한다.
5. Joint calibration을 더 큰 untouched family holdout에서 다시 검증한다.
6. Selector는 현재 holdout에 맞춰 재튜닝하지 말고 구조적으로 재설계한다.
7. 위 blocker가 실제로 변경된 뒤 하나의 새로운 외부 benchmark를 동결하고 Phase 16을 다시 실행한다.

현재 `instance_holdout`과 `family_holdout` 결과는 이미 최종 판정에 사용됐다. 이후 모델·threshold·utility weight를 이 결과에 맞춰 변경하고 같은 holdout으로 성공을 주장해서는 안 된다.

## 10. 구현 및 산출물

Phase 16은 다음 결과를 생성한다.

```text
phase16-run/
  data/final/metric_summary.parquet
  data/final/criterion_results.parquet
  data/final/recommendations.parquet
  final_assessment.json
  assessment_registry.json
  validation.json
  manifest.json
  state.json
  progress.jsonl
```

핵심 파일의 역할은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `metric_summary.parquet` | 31개 원시 metric, threshold, 방향, 통과 여부 |
| `criterion_results.parquet` | 10개 최종 기준의 상태와 실패 metric |
| `final_assessment.json` | 최종 verdict, blocker, 권고 |
| `assessment_registry.json` | 고정 threshold와 판정 계약 |
| `validation.json` | 입력·평가·재계산 검증 결과 |
| `manifest.json` | 산출물 SHA-256과 재개 계약 |

CLI 실행 형식은 다음과 같다.

```bash
python -m uriel_v2.probabilistic_lab phase16 \
  --phase6 artifacts/probabilistic/PHASE6_RUN \
  --phase7 artifacts/probabilistic/PHASE7_RUN \
  --phase8 artifacts/probabilistic/PHASE8_RUN \
  --phase9 artifacts/probabilistic/PHASE9_RUN \
  --phase10 artifacts/probabilistic/PHASE10_RUN \
  --phase11 artifacts/probabilistic/PHASE11_RUN \
  --phase12 artifacts/probabilistic/PHASE12_RUN \
  --phase13 artifacts/probabilistic/PHASE13_RUN \
  --phase14 artifacts/probabilistic/PHASE14_RUN \
  --phase15 artifacts/probabilistic/PHASE15_RUN
```

완료된 실행은 같은 입력으로 resume했을 때 기존 manifest를 그대로 유지한다. Manifest나 선행 Phase 산출물을 변조하면 hash mismatch로 중단한다.

## 11. 검증 결과

전체 회귀 테스트 결과는 다음과 같다.

```text
107 passed, 10 warnings in 41.35s
```

경고 10건은 기존 multiprocessing fork 관련 경고이며 테스트 실패는 없었다.

검증한 핵심 항목:

- Phase 4–16 end-to-end 실행
- 두 held-out split의 정확한 포함
- threshold registry와 criterion 순서 고정
- 모든 최종 metric의 finite·unique 조건
- 최종 verdict 독립 재계산 일치
- 입력 artifact hash chain 검증
- 완료 실행 resume의 byte-level manifest 보존
- Phase 16 manifest 변조 감지
- CLI의 `phase16` command 노출

## 12. 최종 판정

| 항목 | 최종 결과 |
|---|---|
| 프로젝트 16개 Phase 구현 | 완료 |
| Phase 16 평가 무결성 | PASS |
| 예측 분포 연구 가치 | 있음 |
| 자동 알고리즘 선택 가치 | 검증 실패 |
| 배포 준비 | 아님 |
| 최종 verdict | **PARTIAL_SUCCESS_RESEARCH_ONLY** |

이번 16-Phase 프로젝트는 실패로 폐기할 단계는 아니다. Point prediction, marginal calibration, survival은 실제 후속 연구 가치가 있다. 그러나 현재 결과로 자동 선택을 운영에 적용하면 안 된다.

따라서 이 프로젝트는 **“예측 모델은 보존하고 선택기는 재설계하는 부분 성공”**으로 종료한다. 다음 평가 주기는 failure/domain/joint/selection blocker를 해결하고 새로운 외부 benchmark를 확보한 뒤에만 시작한다.
