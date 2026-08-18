# Uriel v2 — 확률적 알고리즘 예측 모델 기반 구축 Phase 1 보고서

- 작성일: 2026-08-19 (Asia/Seoul)
- 대상 저장소: `cjftya/uriel-v2`
- 구현 commit: `b30cc82f7c9769af49410dc56d3153550ce50c10`
- 기준 commit: `0769cae2f58414ab30c07a675d9ba0d529b36c8f`
- Python: 3.12.13
- NumPy: 2.3.5
- RNG: PCG64
- 최종 판정: **PHASE 1 PASS**

## Executive Summary

장기 목표인

```text
Problem Structure
+ Random Mechanism
+ Budget
+ Early Trajectory
→ Outcome Distribution
```

을 연구하기 위한 첫 실행 기반을 구현했다.

이번 단계의 판정 대상은 예측 모델 성능이 아니다. 다음 조건을 검증 대상으로 고정했다.

1. 로또 전용 코드와 범용 확률 알고리즘 연구를 분리할 수 있는가?
2. Problem, Algorithm, Randomness, Budget, Run, Trace를 고정 공통 schema로 저장할 수 있는가?
3. 동일 seed와 설정에서 과학적 결과를 재현할 수 있는가?
4. process worker, checkpoint/resume, Parquet 저장, 데이터 품질 검사가 한 파이프라인으로 동작하는가?
5. 5%·10%·20% early trajectory feature를 최종 결과와 분리해 생성할 수 있는가?

모든 조건이 통과했다.

- synthetic problem 24개
- 72 runs
- 공통 trace 504개
- early-trajectory feature 216개
- 실행 실패 0개
- duplicate, missing trace, NaN/Inf, orphan trace 0개
- 신규 테스트 7개 및 전체 78개 테스트 통과
- 독립적으로 두 번 실행한 stochastic 결과가 동일

따라서 **Phase 1 기반 구축은 PASS**로 판정한다.

단, 이번 결과로 문제 구조에 따른 알고리즘 선택 가능성이나 성능 예측 signal을 주장할 수는 없다. 현재 알고리즘은 IID Monte Carlo와 Random Search 두 개뿐이고, 각각 서로 다른 domain에 배정돼 직접 경쟁하지 않는다. 이번 72-run은 계획서의 10,000-run Pilot이 아니라 그 Pilot을 안전하게 실행하기 위한 smoke validation이다.

## 1. 구현 범위

### 1.1 격리된 패키지

범용 연구 코드는 `src/uriel_v2/probabilistic_lab/`에 분리했다. 기존 Lotto CLI와 실험 코드에는 의존하지 않는다.

새 CLI는 다음과 같다.

```bash
python -m uriel_v2.probabilistic_lab pilot ...
python -m uriel_v2.probabilistic_lab validate RUN_DIRECTORY
```

패키지 entry point도 별도로 추가했다.

```text
uriel-probabilistic
```

### 1.2 공통 schema

다음 객체를 고정 schema로 정의했다.

| 객체 | 핵심 내용 |
|---|---|
| `ProblemSpec` | family, domain, dimension, sparsity, noise, entropy, conditioning, multimodality 등 |
| `AlgorithmSpec` | algorithm, family, random mechanism, version, configuration |
| `BudgetSpec` | budget type, total, checkpoint fractions |
| `JobSpec` | problem + algorithm + seed + RNG + budget |
| `RunResult` | quality, runtime, failure, target, first-passage, stagnation |
| `TracePoint` | step, objective, best, improvement, variance, entropy, diversity, distance |

알고리즘 고유 필드는 공통 column을 추가하지 않고 canonical JSON인 `extension_json`에 저장한다. 알고리즘 configuration도 `algorithm_config_json`에 분리했다.

schema version은 `1.0.0`으로 기록한다.

### 1.3 결정적 run identity

`run_id`는 다음 정보의 canonical JSON SHA-256에서 생성한다.

```text
schema version
problem metadata
algorithm metadata/configuration
seed
RNG algorithm/version
budget/checkpoints
```

runtime이나 실행 시각은 ID에 포함하지 않는다. 따라서 동일 과학 조건은 동일 ID를 만들며, seed·budget·algorithm configuration 중 하나라도 달라지면 다른 ID를 만든다.

### 1.4 Worker와 checkpoint

worker는 `spawn` 기반 process pool을 사용한다.

```text
START
→ problem/algorithm 확인
→ PCG64 초기화
→ algorithm 실행
→ common trace 생성
→ typed result/failure 생성
→ checkpoint.jsonl append + flush
→ DONE
```

재실행 시 이미 완료된 `run_id`는 건너뛴다. 현재 job set과 checkpoint가 다르면 재개하지 않고 오류를 낸다.

### 1.5 Failure schema

정상 실행과 목표 품질 도달 여부를 분리했다.

- `status=SUCCESS`: 실행 자체가 정상 완료
- `target_reached`: 문제별 목표 품질 도달
- `failure`: 실행 실패
- `failure_type`: `FAIL_RNG`, `FAIL_NUMERIC`, `FAIL_MEMORY`, `FAIL_EXECUTION`, 향후 `FAIL_TIMEOUT`
- `failure_time`: 실패 시점

예외를 버리지 않고 연구 데이터로 남기도록 했다.

## 2. Synthetic 문제

### 2.1 Sampling

| Family | 변화 축 |
|---|---|
| Gaussian mean | dimension, scale/noise |
| Student-t mean | dimension, scale, degrees of freedom, heavy tail |
| Mixture mean | dimension, scale, component separation, multimodality |

Monte Carlo는 sample mean을 추정하며 true mean과의 RMSE를 objective로 사용한다.

### 2.2 Optimization

| Family | 변화 축 |
|---|---|
| Sphere | dimension, rotation, ill-conditioning |
| Rastrigin | dimension, rotation, ill-conditioning, multimodality |
| Rosenbrock | dimension, rotation, valley/ruggedness |

Random Search는 problem bound 내부에서 IID uniform proposal을 생성한다.

각 problem은 별도 `problem_seed`로 shift/rotation을 결정한다. run seed는 알고리즘 stochastic trajectory만 결정한다.

## 3. Trace와 feature

### 3.1 Budget checkpoint

의도한 checkpoint는 다음과 같다.

```text
1%, 2%, 5%, 10%, 20%, 50%, 100%
```

정수 budget으로 올림 처리하므로 저장되는 `budget_fraction`은 실제 관측 step/total이다. 작은 budget에서도 checkpoint가 중복되지 않도록 integer step을 unique/monotone하게 만든다.

### 3.2 Common trace

각 checkpoint에서 다음을 기록했다.

```text
objective
best_so_far
improvement
improvement_rate
variance
entropy
diversity
distance_to_best
distance_to_target
failure_signal
elapsed_time
```

Monte Carlo extension에는 estimate, standard error, sample count를 저장한다. Random Search extension에는 trial count와 coverage proxy를 저장한다.

### 3.3 Early trajectory feature

5%, 10%, 20% cutoff별로 다음 feature를 생성했다.

```text
objective_last
best_so_far
improvement_sum
improvement_slope
variance_mean
entropy_mean
diversity_mean
autocorrelation_lag1
transition_magnitude_mean
stagnation_fraction
failure_signal_max
```

최종 결과 column을 사용하지 않고 해당 cutoff까지의 trace만 사용한다.

## 4. 저장 구조

한 실행 디렉터리에 다음 구조를 생성했다.

```text
data/
  problems/problem_metadata.parquet
  runs/runs.parquet
  traces/common/trace_common.parquet
  features/problem_features.parquet
  features/trajectory_features.parquet
checkpoint.jsonl
config.json
summary.json
validation.json
uriel.log
```

대규모 데이터는 git에 포함하지 않으며 seed와 config로 재생성한다.

## 5. Smoke Pilot 조건

| 항목 | 값 |
|---|---:|
| master seed | `20260819` |
| problem families | 6 |
| family별 instances | 4 |
| 총 problem instances | 24 |
| problem별 seeds | 3 |
| 총 runs | 72 |
| Monte Carlo budget | 4,096 samples |
| Random Search budget | 2,048 evaluations |
| workers | 4 |
| git commit | `43dd7500...` |
| job ID set SHA-256 | `a28200e964362f9fd7fbd6c813190bdef8fad45a9647dcccac729b47050699ac` |

실행 명령은 다음과 같다.

```bash
python -m uriel_v2.probabilistic_lab pilot \
  --instances-per-family 4 \
  --seeds 3 \
  --monte-carlo-budget 4096 \
  --random-search-budget 2048 \
  --workers 4 \
  --output artifacts/probabilistic
```

최종 provenance가 기록된 실행은 다음이다.

```text
artifacts/probabilistic/20260819-073948-probabilistic-pilot
```

## 6. 데이터 품질 결과

| 검사 | 결과 |
|---|---:|
| problem rows | 24 |
| run rows | 72 |
| successful runs | 72 |
| failed runs | 0 |
| common trace rows | 504 |
| trajectory feature rows | 216 |
| duplicate run | 0 |
| duplicate problem | 0 |
| missing trace | 0 |
| orphan trace | 0 |
| missing final checkpoint | 0 |
| invalid numeric NaN/Inf | 0 |
| invalid extension JSON | 0 |
| 최종 validation | **PASS** |

216 feature rows는 `72 runs × 3 cutoffs`와 정확히 일치한다.

## 7. 재현성 검증

동일 config를 서로 다른 실행 디렉터리에서 두 번 실행했다.

비교에서 wall-clock 관련 column인 `runtime`, `failure_time`, `elapsed_time`만 제외했다. 이 값은 시스템 scheduling에 따라 달라지므로 stochastic 결과가 아니다.

| 비교 대상 | 결과 | Scientific SHA-256 |
|---|---|---|
| run scientific columns | byte-equivalent table | `57be19adc248e0c71d668b4bb344a4a0434339e42ec49eef31cf9d9a270d815b` |
| trace scientific columns | byte-equivalent table | `4dde8225c999bbf2d7c1cec248600bd493085a082f90efafe337f0a254d911f6` |

동일 seed와 설정에서 quality, objective trajectory, variance, entropy, diversity, first-passage 및 extension 결과가 모두 동일했다.

checkpoint 재개 테스트에서는 완료된 checkpoint를 변경하지 않고 0개 job만 실행하는 것도 확인했다.

## 8. Smoke 결과

이번 quality는 임시 공통 변환인 다음 식이다.

```text
quality = 1 / (1 + objective)
```

모델 학습 전에 family/domain별 utility normalization을 다시 정의해야 하므로, 아래 결과는 algorithm 우열 판정이 아니라 데이터가 의도한 난이도 차이를 담는지 확인하는 용도다.

| Algorithm | Problem family | Runs | Mean quality | SD | Target rate | Failure |
|---|---|---:|---:|---:|---:|---:|
| Monte Carlo | Gaussian mean | 12 | 0.9757 | 0.0212 | 75.0% | 0% |
| Monte Carlo | Student-t mean | 12 | 0.9787 | 0.0116 | 100.0% | 0% |
| Monte Carlo | Mixture mean | 12 | 0.9648 | 0.0242 | 100.0% | 0% |
| Random Search | Sphere | 12 | 0.3319 | 0.4038 | 25.0% | 0% |
| Random Search | Rastrigin | 12 | 0.1522 | 0.2683 | 0.0% | 0% |
| Random Search | Rosenbrock | 12 | 0.2527 | 0.4115 | 16.7% | 0% |

Optimization dimension별 quality는 다음처럼 감소했다.

| Dimension | Runs | Mean quality | Target rate |
|---:|---:|---:|---:|
| 2 | 9 | 0.8344 | 55.6% |
| 5 | 9 | 0.1403 | 0.0% |
| 10 | 9 | 0.0013 | 0.0% |
| 20 | 9 | 0.0063 | 0.0% |

Sampling의 20차원도 mean quality `0.9503`, target rate `66.7%`로 2·5·10차원의 target rate 100%보다 어려웠다.

이는 dimension, conditioning, multimodality 같은 problem feature가 실행 결과 분포에 반영될 수 있음을 보여주는 smoke evidence다. 다만 표본이 작고 algorithm 경쟁이 없으므로 예측 signal이나 통계적 일반화 결과로 해석하지 않는다.

## 9. 테스트

신규 테스트 7개를 추가했다.

- budget checkpoint monotonicity
- synthetic problem 결정성 및 ID uniqueness
- Monte Carlo same-seed scientific reproduction
- Random Search same-seed scientific reproduction
- seed 변화에 따른 run ID 변화
- multi-process worker, Parquet, checkpoint/resume, validation 통합
- changed configuration resume 차단

전체 결과:

```text
78 tests passed
```

기존 Lotto/seed/motif/Top30 테스트를 포함한 전체 회귀 테스트가 통과했다.

## 10. 실패 사례와 한계

### 10.1 failure 분포가 아직 없다

72개 run이 모두 정상 종료돼 failure model이나 survival model을 검증할 데이터가 없다. timeout, numeric divergence, infeasible, artificial stagnation benchmark를 이후 의도적으로 포함해야 한다.

### 10.2 random mechanism 다양성이 아직 없다

두 구현 모두 Independent Sampling 계열이다. Q2인 “알고리즘 이름보다 random mechanism이 중요한가”는 아직 검사할 수 없다.

### 10.3 algorithm selection 데이터가 아니다

Monte Carlo는 sampling problem, Random Search는 optimization problem에만 적용됐다. 같은 problem에서 둘을 비교하지 않았으므로 regret, Top-k coverage, expected utility를 계산해서는 안 된다.

### 10.4 quality scale은 임시다

`1/(1+objective)`는 storage pipeline 검증용이다. objective scale이 다른 family 사이에서 동일 utility를 뜻하지 않는다. 본 Pilot 전에 normalized regret, target-relative quality 또는 family-specific calibrated utility를 명시해야 한다.

### 10.5 runtime 측정 범위가 너무 짧다

개별 실행이 수 ms 수준이라 process scheduling과 logging 영향이 크다. survival/runtime 모델링에는 수 초 이상 budget과 timeout censoring이 필요하다.

### 10.6 과학적 재현성과 wall-clock 재현성은 다르다

동일 seed에서 stochastic 결과는 동일하지만 실제 runtime과 elapsed trace는 동일할 수 없다. 보고서와 validator에서 둘을 분리한다.

## 11. 계획 대비 판정

| 계획 항목 | 상태 |
|---|---|
| 공통 data schema | 완료 |
| Runner / Worker | 완료 |
| RNG / seed control | 완료 |
| Budget checkpoints | 완료 |
| Common trace | 완료 |
| Checkpoint / resume | 완료 |
| Detailed logging | 완료 |
| Parquet dataset | 완료 |
| Monte Carlo pipeline | 완료 |
| Random Search pipeline | 완료 |
| Synthetic benchmark 최소형 | 완료 |
| Data quality validator | 완료 |
| 5/10/20% trajectory feature | 완료 |
| 10,000-run Pilot | 미실행 — Phase 2 algorithm coverage 이후 |
| Distributional/failure/survival model | 미구현 — 데이터 확보 이후 |
| Algorithm selection/OOD | 미구현 — 직접 경쟁 데이터 확보 이후 |

## 12. 다음 단계

다음 구현은 algorithm 수 자체보다 동일 problem에서 random mechanism을 직접 비교할 수 있게 만드는 순서가 적합하다.

1. sampling domain에 RQMC를 추가해 IID Monte Carlo와 paired comparison
2. optimization domain에 CMA-ES를 추가해 Random Search와 paired comparison
3. 동일 problem/seed block에서 paired seed 정책 확정
4. domain별 quality normalization 및 utility 사전등록
5. timeout/stagnation/numeric failure를 발생시키는 stress benchmark 추가
6. 10 seeds로 확대하고 variance-adaptive 30/100 seed 확장 규칙 구현
7. 그 이후에만 10,000-run Pilot 실행
8. problem-instance group split을 고정한 baseline EPM 구축

RQMC와 CMA-ES를 먼저 추가하면 Independent Sampling, Structured Sampling, Adaptive Distribution이라는 세 random mechanism을 같은 domain 내부에서 비교할 수 있다. 이때부터 Q1/Q2의 최소 검증이 가능해진다.

## 13. 최종 결론

이번 단계에서 좋은 예측 모델을 만들지는 않았다. 대신 앞으로 algorithm과 stochastic process를 추가해도 핵심 schema, run identity, worker, trace, checkpoint, Parquet 저장, early feature, validation을 다시 만들지 않아도 되는 기반을 구축했다.

따라서 Phase 1은 **PASS**다.

다음 Phase의 성공 기준은 단순히 RQMC와 CMA-ES가 실행되는 것이 아니다. 동일 problem에서 서로 다른 random mechanism의 결과 분포가 비교 가능하고, paired seed 반복과 quality normalization이 모델 학습에 사용할 수 있는 수준으로 검증되는 것이어야 한다.
