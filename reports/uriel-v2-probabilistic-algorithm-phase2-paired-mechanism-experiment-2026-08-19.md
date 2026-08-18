# Uriel v2 확률적 알고리즘 Phase 2 — Paired Mechanism Experiment

- 실행일: 2026-08-19 (Asia/Seoul)
- 판정: **PHASE_2_PASS**
- 구현 commit: `91d426b3ed1242ec8148824ce518d3d32bee3d17`
- 확정 실행: `artifacts/probabilistic/20260819-081638-probabilistic-phase2`
- 독립 재실행: `artifacts/probabilistic/reproducibility/20260819-081819-probabilistic-phase2`

## 1. 결론

Phase 2의 공통 목표였던 RQMC와 CMA-ES 구현, paired-seed 비교, 사전 판정 기준, 비교용 Parquet 산출물, 재현성 검사를 완료했다. 확정 실행은 48개 problem instance, 960개 run, 480개 paired comparison으로 구성되며 실행 실패는 0건이다.

사전 등록한 `MECHANISM_SIGNAL` 기준은 두 비교에서 모두 충족했다.

| 비교 | 문제 수 | paired run | 평균 품질 차이 | problem bootstrap 95% CI | 단측 Wilcoxon p | challenger 승률 | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| RQMC − IID Monte Carlo | 24 | 240 | +0.031253 | [0.023640, 0.039089] | 5.96e-08 | 99.17% | MECHANISM_SIGNAL |
| CMA-ES − Random Search | 24 | 240 | +0.347145 | [0.180933, 0.522687] | 7.48e-05 | 93.75% | MECHANISM_SIGNAL |

이 결과는 **현재 synthetic benchmark와 동일 평가 예산 안에서 두 challenger가 각 baseline보다 유리했다**는 뜻이다. 알고리즘마다 하나의 random mechanism만 관측했으므로, 아직 알고리즘 효과와 메커니즘 효과를 인과적으로 분리하거나 새 알고리즘 선택 능력을 입증한 결과는 아니다.

## 2. 목적과 가설

Phase 2의 목적은 다음 두 질문을 paired 실험으로 검증할 수 있는 최소 기반을 만드는 것이었다.

1. Independent Sampling보다 Structured Sampling이 동일 sampling 문제·seed·예산에서 더 높은 최종 품질을 보이는가?
2. Independent Sampling 기반 Random Search보다 Adaptive Distribution 기반 CMA-ES가 동일 최적화 문제·seed·평가 예산에서 더 높은 최종 품질을 보이는가?

주 지표는 다음과 같이 정의했다.

\[
\Delta Q = Q_{challenger} - Q_{baseline}, \qquad Q=\frac{1}{1+objective}
\]

추론 단위는 seed run이 아니라 **각 problem instance에서 10개 paired seed 차이를 평균한 값**이다. 같은 문제의 seed 반복을 독립 표본처럼 취급하는 의사 반복을 피했다.

## 3. 구현 범위

### 3.1 추가 알고리즘

- `rqmc_sobol`: SciPy scrambled Sobol, LMS+shift scrambling, Gaussian·Student-t·mixture inverse-CDF 변환
- `cma_es`: 평균, 공분산, step size를 적응시키는 CMA-ES 구현

두 알고리즘은 기존 `AlgorithmSpec → AlgorithmOutput` 계약과 공통 run/trace schema를 그대로 사용한다. CMA-ES 고유 상태인 generation, sigma, covariance eigenvalue, condition, elite spread, stagnation과 RQMC 고유 상태인 discrepancy proxy 등은 extension field에만 기록한다.

### 3.2 비교·판정 계층

- 같은 `problem_id`, `seed`, `budget_type`, `budget`을 갖는 pair 생성
- run 단위 paired 결과와 problem 단위 seed 집계 결과를 각각 Parquet으로 저장
- 품질 차이, 목표 도달 차이, objective ratio, runtime ratio, pair 내부 oracle regret 계산
- problem-instance bootstrap 10,000회 및 단측 Wilcoxon signed-rank test
- 실행 전에 preregistration JSON을 저장하고 SHA-256을 config에 고정

### 3.3 데이터 정합성 수정

최초 확인 실행에서 Rosenbrock의 `ill_conditioned` variant가 metadata에는 condition number를 기록하지만 실제 좌표 scaling에는 반영하지 않는 문제를 발견했다. objective transformation에 축별 scale을 적용하도록 수정하고 테스트를 추가한 뒤, 이전 실행을 폐기하고 commit된 코드로 확정 실험을 다시 수행했다. 본 보고서의 모든 수치는 수정 후 확정 실행만 사용한다.

## 4. 실험 설계

| 항목 | 설정 |
|---|---|
| master seed | 20260820 |
| problem family | Gaussian mean, mixture mean, Student-t mean, Sphere, Rosenbrock, Rastrigin |
| instance | family당 8개, 총 48개 |
| dimension | 2, 5, 10, 20 |
| 반복 | problem-algorithm당 paired seed 10개 |
| sampling 예산 | 4,096 samples |
| optimization 예산 | 4,096 objective evaluations |
| 총 run | 960 |
| 총 pair | 480 |
| trace / trajectory feature | 6,720 / 2,880 |
| worker | 8 |
| Python / NumPy | 3.12.13 / 2.3.5 |

Preregistration SHA-256은 `877815b4af7a05c9f18386216c8d31378fe27598d4600f9d790acf468ece4f23`, 전체 job ID SHA-256은 `73bc42f005eb5e0bb808229dcb0c11f5a75894df59e495e59415d0763326eeae`다.

`MECHANISM_SIGNAL`은 다음 조건을 모두 만족할 때만 부여했다.

- problem instance 8개 이상
- problem-weighted 평균 품질 차이 > 0
- problem bootstrap 95% CI 하한 > 0
- 단측 Wilcoxon p ≤ 0.05
- challenger seed-pair 승률 ≥ 60%
- 모든 problem family의 평균 품질 차이 > 0

## 5. 결과

### 5.1 RQMC 대 IID Monte Carlo

RQMC는 240개 pair 중 238개에서 승리했고, Student-t 5차원 instance의 2개 seed에서만 패했다. 두 패배의 품질 차이는 각각 -0.005885, -0.004228이었다.

| family | pair | 평균 품질 차이 | challenger 승/패 | baseline 목표 도달률 | challenger 목표 도달률 | runtime 중앙비 |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian mean | 80 | +0.026674 | 80 / 0 | 86.25% | 100% | 2.45× |
| Mixture mean | 80 | +0.045533 | 80 / 0 | 100% | 100% | 3.03× |
| Student-t mean | 80 | +0.021552 | 78 / 2 | 96.25% | 100% | 4.53× |

전체 pair에서 목표 도달률은 IID 94.17%, RQMC 100%였다. RQMC의 problem-weighted 평균 log10 objective gain은 1.978이고 pair 내 평균 oracle regret는 IID 0.031295, RQMC 0.000042였다.

동일 sample 수에서도 RQMC의 wall-clock runtime 중앙비는 IID 대비 3.04배였다. 차원이 커질수록 중앙비는 2차원 2.20배에서 20차원 5.39배로 증가했다.

### 5.2 CMA-ES 대 Random Search

CMA-ES는 240개 pair 중 225개에서 승리했다. 15개 패배는 모두 Rastrigin에서 발생했으며, 그중 14개는 2차원 instance였다.

| family | pair | 평균 품질 차이 | challenger 승/패 | baseline 목표 도달률 | challenger 목표 도달률 | runtime 중앙비 |
|---|---:|---:|---:|---:|---:|---:|
| Rastrigin | 80 | +0.017346 | 65 / 15 | 0% | 3.75% | 15.59× |
| Rosenbrock | 80 | +0.338712 | 80 / 0 | 22.50% | 47.50% | 18.54× |
| Sphere | 80 | +0.685378 | 80 / 0 | 25.00% | 100% | 18.13× |

전체 pair에서 목표 도달률은 Random Search 15.83%, CMA-ES 50.42%였다. problem-weighted 평균 log10 objective gain은 8.085이고 pair 내 평균 oracle regret는 Random Search 0.364756, CMA-ES 0.017611이었다.

문제 구조별 이질성은 명확했다.

| dimension | 평균 품질 차이 | CMA-ES 승률 | runtime 중앙비 |
|---:|---:|---:|---:|
| 2 | **-0.054127** | 76.67% | 21.96× |
| 5 | +0.589702 | 98.33% | 20.32× |
| 10 | +0.493299 | 100% | 16.47× |
| 20 | +0.359705 | 100% | 14.20× |

특히 2차원 Rastrigin에서는 Random Search가 평균적으로 더 좋았다. 이는 CMA-ES의 전체 평균 우세를 “항상 우월”로 해석할 수 없으며, dimension과 multimodality 같은 problem structure가 선택 변수로 필요하다는 직접적인 사례다.

동일 objective evaluation 수에서도 CMA-ES의 wall-clock runtime 중앙비는 Random Search 대비 17.48배였다. 따라서 runtime penalty가 큰 utility에서는 품질 우세가 자동으로 최종 선택 우세가 되지 않는다.

## 6. 데이터 품질과 재현성

확정 실행의 자동 검사는 모두 통과했다.

- 48 problems, 960 successful runs, 0 failures
- 480/480 pair coverage
- 문제별 seed 반복 10개 완전성 확인
- 비교용 숫자 필드 NaN·Inf 없음
- 6,720 common trace와 2,880 trajectory feature 생성
- 4개 알고리즘과 3개 random mechanism 존재
- validation issue 0건

같은 commit과 config로 별도 디렉터리에 960개 run을 다시 실행했다. runtime, trace elapsed time, runtime-derived ratio를 제외한 다음 결과는 정렬 후 값과 dtype까지 정확히 일치했다.

- runs의 과학적 결과 필드
- common traces의 상태·objective 필드
- paired run의 품질·목표·objective·winner 필드
- trajectory features 전체

wall-clock 필드는 worker scheduling과 시스템 상태의 영향을 받으므로 재실행 간 동일성을 요구하지 않았다.

검증 명령 결과는 `84 passed`, warning 10건, dependency conflict 0건이었다. warning은 기존 reverse test의 multiprocessing `fork()` deprecation warning이며 Phase 2 계산 실패는 아니다.

## 7. 실패 사례와 해석상 제한

실행 자체의 timeout, divergence, numeric failure는 없었다. 그러나 비교상 패배 사례는 반드시 유지했다.

- RQMC 패배: Student-t mean 5차원 2/240 pair
- CMA-ES 패배: Rastrigin 15/240 pair
- 2차원 optimization 전체 평균에서는 Random Search가 더 우수

현재 결과의 한계는 다음과 같다.

1. 메커니즘별 알고리즘이 하나뿐이라 algorithm identity와 mechanism effect가 혼재한다.
2. synthetic 6개 family, 고정 4,096 예산의 결과이며 budget scaling 일반화를 검증하지 않았다.
3. 같은 seed 숫자를 pair에 사용했지만 알고리즘별 RNG 소비 순서가 달라 완전한 common-random-number 설계는 아니다.
4. runtime은 8-worker 단일 환경 측정치다. 정확한 비용 모델에는 격리된 반복 측정이 필요하다.
5. 두 confirmatory comparison의 p-value에 다중비교 보정을 적용하지 않았다. 다만 두 bootstrap CI 하한도 모두 0보다 컸다.
6. 본 단계에서는 새로운 problem family나 새로운 알고리즘 holdout을 수행하지 않았다.

따라서 이번 판정은 **paired benchmark signal 확인과 Phase 2 인프라 완성**에 한정한다. 확률분포 예측, calibration, 알고리즘 선택 일반화, zero-shot mechanism transfer가 성공했다고 판단하지 않는다.

## 8. 다음 단계 판정

Phase 2 완료 기준을 충족했으므로 다음 단계로 진행할 수 있다. 우선순위는 문제 구조 축과 budget 축을 넓히고, 동일 mechanism 안에 복수 알고리즘을 추가해 algorithm effect와 mechanism-family effect를 분리하는 것이다.

권장 순서는 다음과 같다.

1. budget 1·2·5·10·20·50·100% checkpoint를 독립 job이 아니라 공통 trajectory에서 검증
2. sampling과 optimization에 동일 mechanism의 두 번째 알고리즘을 추가
3. dimension·noise·conditioning·multimodality 축을 균형 설계한 synthetic benchmark 생성
4. problem-instance group split을 고정하고 problem-only baseline 모델 구축
5. runtime penalty를 포함한 utility sensitivity curve 산출
6. 그 후에 early-trajectory feature의 증분 예측력을 측정

## 9. 재현 명령

```bash
PYTHONPATH=. .venv/bin/uriel-probabilistic phase2 \
  --instances-per-family 8 \
  --seeds 10 \
  --master-seed 20260820 \
  --sampling-budget 4096 \
  --optimization-budget 4096 \
  --bootstrap-iterations 10000 \
  --workers 8
```

테스트는 다음 명령으로 재현한다.

```bash
PYTHONPATH=. .venv/bin/pytest
.venv/bin/pip check
```
