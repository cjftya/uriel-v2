# Uriel v2 Answer-Derived Reverse Reconstruction Stage A 결과 보고서

- 작성일: 2026년 8월 13일 (KST)
- 대상 저장소: [cjftya/uriel-v2](https://github.com/cjftya/uriel-v2)
- 반영 브랜치: `main`
- 구현 커밋: [`9f50a1bff1f8b7fa31570309100ba14518cf9816`](https://github.com/cjftya/uriel-v2/commit/9f50a1bff1f8b7fa31570309100ba14518cf9816)
- Pull Request: 없음 — 요청에 따라 `main`에 직접 반영
- 최종 검증 상태: **Ready to share with caveats**

> 이 실험은 이미 알려진 당첨번호를 사용한 **정답 기반 reverse reconstruction**이다. 미래 회차를 예측한 실험이 아니며, 아래 5-hit·6-hit 수치를 predictive performance로 해석할 수 없다.

## 결론

계획서의 Stage A를 완료했다. 1044~1235회 192개 회차에서 각 회차마다 동일한 seed 구간 `[0, 1,000,000)`을 전수 탐색해 총 **192,000,000개 seed 평가**를 수행했다. 회차별 Top-100, 모든 4+/5+/6-hit seed, seed bucket, reconstruction curve, 이론·Monte Carlo random baseline과 실행 로그를 재현 가능한 데이터셋으로 저장했다.

회차 최고 결과는 5-hit 174회, exact 6-hit 18회였다. 192회 모두 5+를 재구성했지만, 1백만 random 조합을 회차마다 탐색할 때도 5+가 하나 이상 나올 이론 확률은 사실상 100%다. exact 6 회차도 관측 18회로 동일 budget Monte Carlo의 95% 범위 14~31회 안에 있다. 후보 전체의 4/5/6-hit 수도 hypergeometric 기대치와 매우 가깝다.

따라서 이번 단계의 판정은 다음과 같다.

> **RECONSTRUCTION DATASET SUCCESS / PREDICTIVE SIGNAL NOT TESTED**

성공한 것은 알려진 정답을 안정적이고 재현 가능하게 seed 구조로 재구성한 데이터셋 구축이다. SplitMix64 seed 공간에서 random baseline을 넘는 예측 신호가 확인됐다는 뜻은 아니다.

## 판정 요약

| 항목 | 결과 | 판정 |
|---|---:|---|
| 완료 회차 | 192 / 192 | 통과 |
| 회차당 seed budget | 1,000,000 | 모든 회차 동일 |
| 총 seed 평가 | 192,000,000 | 계획과 일치 |
| 회차 최고 5-hit | 174회 | random budget에서 자연스러운 결과 |
| 회차 최고 exact 6 | 18회 | Monte Carlo 95% 범위 14~31회 안 |
| 5+ 재구성 | 192회, 100% | random 기대도 사실상 100% |
| Top-K | 19,200행 | 회차당 100개 |
| 모든 4+/5+/6 seed | 266,849행 | 261,288 / 5,540 / 21 |
| worker·chunk 재현성 | 동일 | 통과 |
| 두 전체 실행 결과 | 시간 외 완전 동일 | 통과 |
| 예측 성능 | 평가하지 않음 | reverse와 forward 분리 유지 |

## 실험 범위와 방법

### 고정 Stage A 조건

| 설정 | 값 |
|---|---:|
| 회차 | 1044~1235, 192회 |
| seed 범위 | `[0, 1,000,000)` |
| 회차당 평가 | 1,000,000 |
| 총 평가 | 192,000,000 |
| Top-K | 100 |
| 별도 보존 최소 hit | 4 |
| chunk | 25,000 |
| bucket | 100,000 |
| workers | `auto` → 8 |
| PRNG | 기존 SplitMix64 유지 |
| 보너스 번호 | 점수에서 제외 |

후보 순위는 계획서대로 다음 키를 사용했다.

```text
hits 내림차순
→ positional_mae 오름차순
→ set_distance 오름차순
→ seed 오름차순
```

`set_distance`는 hit 수로 결정되므로 동일 hit 안에서는 실질적으로 `positional_mae`, `seed` 순서가 tie-breaker다. 이 정렬은 worker 수나 chunk 경계와 무관하게 결정적이다.

### 입력 데이터 품질

입력은 저장소의 `lotto.xlsx`, `Lotto` 시트다.

| 검사 | 결과 |
|---|---:|
| 전체 데이터 | 1~1235회, 1,235행 |
| 전체 누락 회차 | 0 |
| 중복 회차 | 0 |
| Stage A 회차 | 1044~1235, 정확히 192행 |
| Stage A 누락 회차 | 0 |
| 번호 개수·중복·1~45 범위 오류 | 0 |
| 보너스 결측 | 0 |
| 입력 SHA-256 | `7efe5e232c7d4ed347b0377726686adad93ebe0581f3d03257cf4e2f83836db4` |

보너스 데이터는 유효하지만 본 실험의 target과 hit 계산에는 사용하지 않았다.

## 구현 내용

### `reverse-batch` CLI

다음 옵션을 갖는 다회차 명령을 추가했다.

```text
--start-round
--end-round
--seed-start
--seed-end
--top-k
--min-hits
--chunk-size
--workers
--bucket-size
```

각 회차 안에서 seed chunk를 `ProcessPoolExecutor`에 분배하고, executor는 회차 사이에서 재사용한다. 모든 seed 결과를 메모리에 쌓지 않고 다음만 유지한다.

- 0~6 hit 분포
- 제한 크기 Top-K heap
- `min-hits` 이상 seed
- bucket별 최고 결과와 4/5/6-hit 집계
- 10K·100K·전체 budget의 누적 최고 결과

### 실행 로그와 내구성

회차 내부 진행률, 평가 수, 현재 최고 hit/seed, 5-hit·6-hit 수, 처리 속도, 회차 ETA와 경과 시간을 콘솔과 `uriel.log`에 기록한다. 회차가 끝날 때마다 CSV를 flush·`fsync`하고 `reverse-progress.json`을 갱신한다.

첫 번째 전체 실행에서는 계산과 JSON 요약은 192회를 포함했지만 최종 CSV 5종이 191회까지만 기록된 불일치를 사후 검산에서 발견했다. 이에 다음을 추가했다.

- 회차별 CSV `fsync`
- 종료 전 파일별 기대 행 수 자동 대조
- 불일치 시 성공 요약을 만들지 않고 실패 처리

보강 후 Stage A 전체를 새 디렉터리에서 다시 실행했다. 두 실행의 시간 필드를 제외한 reconstruction, candidate 분포, random baseline, seed landscape와 특이 회차가 완전히 같았고, 두 번째 실행은 코드 내부 행 수 검증과 별도 검산을 모두 통과했다.

### random baseline과 그래프

- 6/45 exact hypergeometric 확률
- budget 내 하나 이상 4+/5+/6가 나올 확률
- 회차별 최고 hit의 exact IID 분포
- 고정 seed `20260813`, 10,000회 반복 equal-budget Monte Carlo
- 10K·100K·1M budget의 reconstruction 집계
- 0 기준선의 독립 막대 SVG 4종

그래프는 exact 6 rate, 5+ rate, mean best hit, median best positional MAE를 각각 분리했다. 세 budget은 연속 시계열이 아니라 이산 비교점이므로 선 그래프 대신 막대를 사용했다.

## 생성 산출물

최종 실행 디렉터리는 다음과 같다.

```text
outputs/reverse-dataset/20260813-132436-reverse-batch/
```

`outputs/`는 대용량 실행 산출물이므로 Git 추적 대상에서 제외된다.

| 파일 | 데이터 행 | 크기 | 내용 |
|---|---:|---:|---|
| `reverse-rounds.csv` | 192 | 29 KB | 회차별 최고 결과·분포·성능 |
| `reverse-top-k.csv` | 19,200 | 1.06 MB | 회차별 Top-100 |
| `reverse-hit-seeds.csv` | 266,849 | 15.06 MB | 모든 4+/5+/6-hit seed |
| `reverse-seed-buckets.csv` | 1,920 | 122 KB | 회차×100K bucket 집계 |
| `reverse-reconstruction-curve.csv` | 576 | 25 KB | 회차×3 budget 최고 결과 |
| `reverse-summary.json` | — | 179 KB | 전체 요약·baseline·landscape |
| `reverse-progress.json` | — | 2 KB | 완료 상태와 회차 목록 |
| `curve-*.svg` | 4개 | 각 약 3 KB | 독립 reconstruction 그래프 |
| `uriel.log` | — | 481 KB | 전체 진행 로그 |

CSV는 `utf-8-sig`로 저장해 Excel에서도 한글 헤더를 바로 확인할 수 있다.

## 실행 환경과 성능

| 항목 | 값 |
|---|---|
| OS | Linux 6.18.35 x86_64, glibc 2.39 |
| Python | 3.12.13 |
| openpyxl | 3.1.5 |
| 감지 CPU | 9 |
| 사용 worker | 8 |
| 최종 실행 시간 | 169.359초, 약 2분 49초 |
| 전체 처리량 | 1,133,684 seeds/s |

계획서의 10회 × 100K 스모크도 먼저 수행했다. 1,000,000개 평가와 모든 산출물 생성이 정상 완료된 뒤 전체 Stage A를 실행했다.

## 테스트와 검증

최종 자동 테스트는 14/14개 통과했다.

```text
Ran 14 tests in 0.293s
OK
```

추가된 검증은 다음과 같다.

- 단일 `reverse`와 `reverse-batch`의 hit 분포·best·Top-K 일치
- workers 1과 4 결과 일치
- chunk 10·25·100 결과 일치
- Top-K 정렬 안정성
- known exact seed 42 보존
- hypergeometric·maximum 분포 합 1 검증
- CSV·JSON·SVG 전체 산출물 생성 검증
- `lotto.xlsx` 회차 1~1235 연속성 검증
- 최종 CSV 행 수 자동 검증
- Top-K rank 1~100과 정렬 키 독립 검산
- `reverse-rounds.csv` best와 `reverse-top-k.csv` rank 1 연결 검산
- candidate hit 분포 합이 192,000,000인지 검산
- SVG 4종 XML parse와 실제 렌더 확인
- 두 전체 실행의 결정적 결과 일치 검산

## 192회 reconstruction 결과

### 회차별 최고 hit

| 최고 hit | 회차 수 | 비율 |
|---:|---:|---:|
| 5 | 174 | 90.625% |
| 6 | 18 | 9.375% |
| 4 이하 | 0 | 0% |
| 합계 | 192 | 100% |

- 4+ 재구성: 192 / 192회
- 5+ 재구성: 192 / 192회
- exact 6 재구성: 18 / 192회
- exact 6 seed: 21개

### 모든 후보의 4/5/6-hit 분포와 random 기대

두 독립 6/45 조합이 정확히 `k`개 번호를 공유할 확률은 다음과 같다.

\[
P(H=k)=\frac{\binom{6}{k}\binom{39}{6-k}}{\binom{45}{6}}
\]

| hit | 관측 | 이론 기대 | 관측/기대 |
|---:|---:|---:|---:|
| 4 | 261,288 | 262,009.12 | 0.99725 |
| 5 | 5,540 | 5,515.98 | 1.00435 |
| 6 | 21 | 23.57 | 0.89087 |

세 계층 모두 random 기대와 가깝다. 특히 5-hit은 기대보다 0.44% 많고, 4-hit은 0.28% 적다. exact 6는 표본 자체가 21개로 작으며 이론 기대보다 약 2.6개 적다. 이 차이들은 seed 구조의 비무작위 우위를 주장할 근거가 아니다.

### 회차당 1M budget random 기준

| 지표 | 이론값 / Monte Carlo | 관측 |
|---|---:|---:|
| 회차당 5+ 하나 이상 확률 | 99.99999999997% | 100% |
| 회차당 exact 6 하나 이상 확률 | 11.5536% | 9.375% |
| 192회 exact 6 회차 기대 | 약 22.18회 | 18회 |
| Monte Carlo exact 6 회차 평균 | 22.29회 | 18회 |
| Monte Carlo 95% 범위 | 14~31회 | 범위 안 |
| Monte Carlo mean best hit 평균 | 5.1161 | 5.0938 |
| Monte Carlo mean best hit 95% 범위 | 5.0729~5.1615 | 범위 안 |

동일 예산 random 기준에서도 거의 모든 회차의 최고는 5-hit이고 일부 회차만 exact 6가 된다. 관측 분포 `5:174, 6:18`은 이 패턴과 일치한다.

## seed budget 대비 reconstruction curve

| budget / 회차 | 5+ 회차 | 5+ 관측률 | 5+ random 기대 | exact 6 회차 | exact 6 관측률 | exact 6 random 기대 | mean best hit | median best MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 56 | 29.167% | 25.063% | 0 | 0.000% | 0.123% | 4.2917 | 1.1667 |
| 100,000 | 186 | 96.875% | 94.416% | 2 | 1.042% | 1.220% | 4.9792 | 1.0000 |
| 1,000,000 | 192 | 100.000% | ≈100.000% | 18 | 9.375% | 11.554% | 5.0938 | 0.1667 |

10K와 100K에서 관측 5+ 비율이 random 기대보다 각각 4.10%p, 2.46%p 높지만, 이 비교는 단 세 budget의 동일 192회 표본이며 1M에서는 차이가 사라진다. exact 6는 100K와 1M 모두 random 기대보다 낮다. 전체 후보 hit 분포까지 함께 보면 비무작위 lift로 해석할 근거가 없다.

4+는 10K부터 192회 모두 재구성됐다. 단일 random seed의 4+ 확률이 약 0.139%이므로 10K를 탐색하면 4+가 하나 이상 나오는 것이 거의 확실하다. 따라서 4+ 100%도 예측 신호가 아니다.

## Top-K와 seed landscape

### hit 계층

| hit | seed 행 | 고유 seed | 최소 | 중앙 | 최대 | 100K당 밀도 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 261,288 | 229,709 | 7 | 500,756 | 999,997 | 136.0875 |
| 5 | 5,540 | 5,529 | 4 | 492,603 | 999,836 | 2.8854 |
| 6 | 21 | 21 | 13,582 | 586,103 | 991,929 | 0.01094 |

동일 seed가 서로 다른 회차에서 다시 나타날 수 있으므로 4-hit와 5-hit의 행 수가 고유 seed 수보다 크다. exact 6 seed 21개는 모두 고유했다.

### Top-100 분포

| 지표 | 값 |
|---|---:|
| 총 seed | 19,200 |
| 고유 seed | 19,037 |
| 최소 / 최대 | 4 / 999,904 |
| 중앙 | 469,728 |
| P10 / P90 | 87,580 / 888,478 |

Top-K는 seed 공간 전반에 퍼져 있다. 중앙값이 500K보다 낮은 것은 동일 hit·MAE tie에서 작은 seed를 먼저 선택하는 명시적 정렬 규칙의 영향도 받으므로, 이를 낮은 seed 영역의 물리적 집중으로 해석하면 안 된다.

### 100K bucket별 4+ 밀도

| seed bucket | 4-hit | 5-hit | 6-hit | 4+ 합계 | 100K당 밀도 | 전체 4+ 비중 |
|---|---:|---:|---:|---:|---:|---:|
| 0~100K | 26,188 | 575 | 2 | 26,765 | 139.401 | 10.030% |
| 100~200K | 26,245 | 541 | 3 | 26,789 | 139.526 | 10.039% |
| 200~300K | 25,923 | 580 | 1 | 26,504 | 138.042 | 9.932% |
| 300~400K | 26,148 | 555 | 2 | 26,705 | 139.089 | 10.008% |
| 400~500K | 25,940 | 563 | 2 | 26,505 | 138.047 | 9.933% |
| 500~600K | 26,392 | 545 | 2 | 26,939 | 140.307 | 10.095% |
| 600~700K | 25,959 | 528 | 2 | 26,489 | 137.964 | 9.927% |
| 700~800K | 26,393 | 556 | 2 | 26,951 | 140.370 | 10.100% |
| 800~900K | 26,056 | 537 | 2 | 26,595 | 138.516 | 9.966% |
| 900K~1M | 26,044 | 560 | 3 | 26,607 | 138.578 | 9.971% |

최고·최저 4+ 밀도 비율은 1.0174다. 각 bucket이 전체 4+ seed의 9.927~10.100%를 차지해 고정 100K 단위에서 뚜렷한 집중은 보이지 않는다.

## 특이 회차

### exact 6 seed가 있는 18개 회차

| 회차 | exact seed |
|---:|---|
| 1071 | 730,760 |
| 1075 | 50,325 |
| 1081 | 363,397 |
| 1085 | 134,431 |
| 1088 | 991,929 |
| 1115 | 671,646 |
| 1121 | 984,784 |
| 1129 | 13,582 |
| 1143 | 893,708 |
| 1146 | 851,636 |
| 1148 | 586,103; 739,815 |
| 1170 | 586,938; 911,586 |
| 1172 | 443,381 |
| 1183 | 163,185 |
| 1186 | 353,755 |
| 1210 | 228,199; 426,171 |
| 1230 | 180,454 |
| 1231 | 648,058 |

1148, 1170, 1210회는 exact seed가 각각 2개였고 나머지는 1개였다. 한 회차당 exact seed 기대값은 약 0.123개이므로 복수 exact seed는 드물지만, 192회 전체 탐색에서는 가능한 tail event다. 이 회차만 선택해 후속 규칙을 만들면 outcome-based selection bias가 생긴다.

### 5-hit seed가 많은 회차

| 회차 | 5-hit seed 수 | 회차 최고 seed |
|---:|---:|---:|
| 1222 | 45 | 150,001 |
| 1060 | 41 | 473,756 |
| 1173 | 41 | 42,188 |
| 1174 | 41 | 71,647 |
| 1102 | 40 | 202,271 |

이 회차들은 기술적으로는 landscape 검토 대상이지만, 정답을 보고 선별된 집합이다. 별도 forward 성공 사례나 seed 구조의 일반화 증거로 사용할 수 없다.

## 해석과 한계

### 확정할 수 있는 것

- 192개 회차를 동일 seed budget으로 안정적으로 처리했다.
- Top-K와 모든 4+/5+/6-hit seed를 결정적으로 재현할 수 있다.
- worker 수와 chunk 크기가 결과를 바꾸지 않는다.
- seed 공간의 100K bucket별 4+ 밀도는 매우 고르게 분포했다.
- 4/5/6-hit 수와 회차 최고 분포는 random baseline과 일치한다.

### 확정할 수 없는 것

- reverse seed에서 미래 회차를 예측할 수 있는 구조가 존재하는지
- exact 6가 나온 회차나 seed가 다음 회차에서도 유효한지
- seed delta, modulo, bit pattern, Hamming distance가 forward 일반화를 제공하는지
- 더 큰 seed 범위가 예측력으로 이어지는지

이번 결과는 정답을 사용해 후보를 정의했다. 따라서 reverse 데이터셋에서 발견한 규칙은 그 자체로 in-sample이며, forward 규칙을 사전에 동결하고 별도 구간에서 검증하기 전에는 예측 성능 주장을 할 수 없다.

## 다음 단계 후보

1. **Development 내부 구조 분석**
   - Top-K seed delta와 회차 간 이동
   - modulo 계열과 bit 분포
   - Hamming distance
   - hit 계층별 군집 중심·분산
   - exact 회차를 사후 선택하지 않은 전체 192회 기준 분석

2. **가설과 선택 규칙 사전 고정**
   - `이전 회차 reverse seed set → 다음 회차 forward seed set` 변환을 명시
   - 후보 수와 계산 budget을 random baseline과 동일하게 유지
   - score·임계값·tie-breaker를 결과 확인 전에 고정

3. **Stage B 진입 조건 정의**
   - Stage A 데이터로 adaptive range의 평가 기준을 먼저 문서화
   - 단일 exact 사례가 아니라 전체 회차 기준의 사전 정의 지표 사용
   - Stage B와 Stage A 성능표를 분리

4. **봉인 구간 유지**
   - Historical Reference 852~1043은 구조 설명 보조로만 사용
   - Locked Holdout 660~851과 Additional Blind 468~659는 규칙 동결 전까지 열지 않음
   - reverse reconstruction과 forward prediction을 같은 성과표에 합치지 않음

5. **실행 복구 기능 보강**
   - 현재는 회차별 progress와 durable CSV를 남기지만 자동 resume는 하지 않는다.
   - 더 큰 Stage B/C 탐색 전에 완료 회차를 안전하게 이어받는 `--resume`을 추가하는 것이 적절하다.

## 변경된 파일

| 파일 | 변경 |
|---|---|
| `src/uriel_v2/reverse_batch.py` | 배치 실행, 집계, CSV/JSON, progress, 자체 행 수 검증 |
| `src/uriel_v2/baselines.py` | hypergeometric·maximum·Monte Carlo baseline |
| `src/uriel_v2/charts.py` | dependency-free SVG budget 막대 그래프 |
| `src/uriel_v2/cli.py` | `reverse-batch` 명령과 옵션 |
| `src/uriel_v2/models.py` | reverse distance·bias·deviation 필드 |
| `src/uriel_v2/reverse.py` | batch와 공유하는 결정적 match 생성·정렬 |
| `tests/test_reverse_batch.py` | batch·worker·chunk·exact·baseline·산출물 테스트 |
| `tests/test_data.py` | 전체 회차 연속성 테스트 |
| `README.md` | v0.2 사용법과 산출물 문서 |
| `pyproject.toml` | 버전 0.2.0 |
| `src/uriel_v2/__init__.py` | 런타임 버전 0.2.0 |

## 재현 명령

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m uriel_v2 reverse-batch \
  --data lotto.xlsx \
  --start-round 1044 \
  --end-round 1235 \
  --seed-start 0 \
  --seed-end 1000000 \
  --top-k 100 \
  --min-hits 4 \
  --chunk-size 25000 \
  --bucket-size 100000 \
  --workers auto

python -m unittest discover -s tests -v
```

실행 결과 폴더는 시각·환경에 따라 이름이 달라지지만, 동일 입력 파일·SplitMix64·옵션을 사용하면 시간과 처리량을 제외한 reconstruction 결과는 동일해야 한다.

## 최종 검증 평가

**Ready to share with caveats**로 판정한다.

- 방법론: 계획서의 fixed Stage A 조건과 일치
- 데이터: 회차·번호 완전성 검증 통과
- 계산: CSV·JSON·독립 집계가 일치
- 재현성: worker·chunk·전체 2회 실행 결과 일치
- 시각화: 0 기준선, 단일 축, 독립 그래프, 라벨 검증 통과
- 필수 caveat: 정답 기반 reconstruction이며 forward predictive evidence가 아님

현재 남은 blocker는 없다. 후속 구조 분석에서만 봉인 구간 유지, 사전 규칙 고정, 동일 budget random 비교가 다시 필요하다.
