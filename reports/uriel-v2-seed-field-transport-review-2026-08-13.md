# Uriel v2 Seed Field Transport 검증 보고서

- 작성일: 2026-08-13 (Asia/Seoul)
- 실험 단계: Stage B — answer-derived seed landscape의 다음 회차 운반 가능성 검정
- 대상 저장소: `cjftya/uriel-v2`
- 판정: **실패 — forward signal 미확인, Locked/Blind 봉인 유지**

## 기술 요약

이번 실험은 회차별로 하나의 대표 seed를 고르는 대신, 4개 이상 당첨번호가 일치한 모든 seed를 여러 좌표계의 확률장(field)으로 바꾼 뒤 그 장의 이동을 다음 회차로 외삽하는 비정상적 가설을 검정했다. 정수축, Gray code, 20-bit 반전, 7-bit 회전, 짝·홀 비트 분리, 소수 모듈러 좌표를 함께 사용하고 persistence, EWMA, analog, circular shift, XOR, equal-weight ensemble을 고정했다.

결과는 예측력을 지지하지 않았다. 주 지표인 ensemble Top-1,000의 5+ seed는 Development에서 관측 3개/기대 5.6125개(lift 0.535, 단측 `p=0.9160`), Historical에서 관측 5개/기대 5.4481개(lift 0.918, `p=0.6327`)였다. Top-1,000의 4+도 각각 lift 1.060(`p=0.1731`)과 1.015(`p=0.4098`)로 우연 변동 범위였다.

1235회까지만 사용해 산출한 1236회 ensemble 1순위는 seed `163185`, 번호 `4, 15, 17, 23, 27, 36`이었다. 사용자가 알려준 실제 당첨번호 `12, 18, 21, 29, 34, 38`과의 적중은 0개였다. Ensemble Top-100 최고는 3개 적중, 4+는 0건이었고 deterministic random Top-100도 최고 3개·4+ 0건이었다. 따라서 1236회에서도 구별되는 우위를 보이지 않았다.

이 결과는 “엉뚱한 가설을 시도할 가치가 없다”는 뜻이 아니다. 다만 **이 특정 가설이 현재의 엄격한 walk-forward 검증을 통과하지 못했다**는 뜻이다. 성과를 만들기 위해 설정을 사후 조정하거나 Locked/Blind 구간을 열지 않았다.

## 핵심 결과

### Ensemble budget별 결과

관측치는 선택된 seed 수의 합이다. 기대값과 단측 p-value는 각 회차의 실제 qualifying seed 개수를 조건으로 한 hypergeometric null을 10,000회 시뮬레이션해 계산했다.

| Cohort | Budget | 4+ 관측/기대 | 4+ lift | p | 5+ 관측/기대 | 5+ lift | p | exact 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | 10 | 0 / 2.6807 | 0.000 | 1.0000 | 0 / 0.0567 | 0.000 | 1.0000 | 0 |
| Development | 100 | 25 / 26.7689 | 0.934 | 0.6604 | 1 / 0.5509 | 1.815 | 0.4211 | 0 |
| Development | 1,000 | 283 / 267.1043 | 1.060 | 0.1731 | 3 / 5.6125 | 0.535 | 0.9160 | 0 |
| Development | 10,000 | 2,744 / 2,669.2052 | 1.028 | 0.0727 | 63 / 55.5920 | 1.133 | 0.1698 | 1 |
| Historical | 10 | 4 / 2.6825 | 1.491 | 0.2855 | 0 / 0.0536 | 0.000 | 1.0000 | 0 |
| Historical | 100 | 29 / 26.7788 | 1.083 | 0.3522 | 1 / 0.5348 | 1.870 | 0.4138 | 0 |
| Historical | 1,000 | 271 / 267.0041 | 1.015 | 0.4098 | 5 / 5.4481 | 0.918 | 0.6327 | 0 |
| Historical | 10,000 | 2,700 / 2,669.2435 | 1.012 | 0.2761 | 41 / 54.1091 | 0.758 | 0.9724 | 0 |

Top-1,000 random null의 95% 구간은 Development 4+ `[236, 299]`, 5+ `[1, 11]`; Historical 4+ `[235, 299]`, 5+ `[1, 10]`이었다. Ensemble 관측치는 모두 이 구간 안에 있다.

### 모델별 Top-1,000 비교

| Cohort | 모델 | 4+ 관측 | lift | p | 5+ 관측 | lift | p | exact 6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Development | persistence | 256 | 0.958 | 0.7621 | 4 | 0.713 | 0.8081 | 1 |
| Development | EWMA | 270 | 1.011 | 0.4358 | 6 | 1.069 | 0.4954 | 0 |
| Development | analog | 283 | 1.060 | 0.1731 | 8 | 1.425 | 0.2053 | 1 |
| Development | shift | 260 | 0.973 | 0.6797 | 3 | 0.535 | 0.9160 | 0 |
| Development | XOR | 277 | 1.037 | 0.2795 | 8 | 1.425 | 0.2053 | 0 |
| Development | ensemble | 283 | 1.060 | 0.1731 | 3 | 0.535 | 0.9160 | 0 |
| Development | random | 241 | 0.902 | 0.9496 | 5 | 0.891 | 0.6619 | 0 |
| Historical | persistence | 270 | 1.011 | 0.4342 | 4 | 0.734 | 0.7947 | 0 |
| Historical | EWMA | 270 | 1.011 | 0.4342 | 3 | 0.551 | 0.9108 | 0 |
| Historical | analog | 268 | 1.004 | 0.4812 | 5 | 0.918 | 0.6327 | 0 |
| Historical | shift | 284 | 1.064 | 0.1590 | 4 | 0.734 | 0.7947 | 0 |
| Historical | XOR | 272 | 1.019 | 0.3873 | 3 | 0.551 | 0.9108 | 0 |
| Historical | ensemble | 271 | 1.015 | 0.4098 | 5 | 0.918 | 0.6327 | 0 |
| Historical | random | 240 | 0.899 | 0.9563 | 5 | 0.918 | 0.6327 | 0 |

어느 학습 모델도 두 코호트에서 같은 방향의 유의한 개선을 재현하지 못했다.

## 흥미롭지만 채택할 수 없는 이상점

### Development 1148회 exact-6

1148회에는 exact-6 seed가 전체 100만 공간에 2개 있었다.

| 모델 | seed | 선택 순위 | 포함 budget | nominal p |
|---|---:|---:|---:|---:|
| persistence | 586103 | 66 | Top-100 | 0.0029 |
| analog | 739815 | 261 | Top-1,000 | 0.0190 |
| ensemble | 739815 | 4,310 | Top-10,000 | 0.1947 |

Persistence Top-100의 exact-6 한 건은 단일 검정만 보면 이례적이다. 그러나 이는 여러 모델·budget·threshold·cohort를 동시에 본 탐색 중 하나이고, 같은 1148회에서 중첩된 seed 선택이 반복 집계됐다. 6개 학습 모델 × 4개 budget × 3개 threshold × 2개 cohort의 144개 관측을 보수적으로 보정하면 `0.0029 × 144 ≈ 0.418`이다. Historical에서 persistence exact-6 재현도 없었다. 따라서 발견이 아니라 **후속 가설 후보**로만 기록한다.

### Historical shift Top-10

Shift는 Historical Top-10에서 4+ 7개/기대 2.6825개(lift 2.610, nominal `p=0.0213`)를 기록했고 그중 한 개는 5-hit였다. Development에서는 4+ 4개/기대 2.6807개(lift 1.492, `p=0.2791`)로 약해졌다. Primary ensemble 결과가 아니며 다중 관측 보정 후 유의하지 않으므로 승격하지 않았다.

이 두 이상점은 “무작위처럼 보이는 전체 속에서도 좁은 연산자가 간헐적으로 맞을 수 있다”는 아이디어를 유지할 근거는 되지만, 다음 회차에 돈을 걸 수 있는 증거는 아니다.

## 1236회 사후 대조

후보 생성 시 사용한 정보는 820–1235회 seed landscape뿐이다. 1236회 정답은 후보 CSV 생성 후 대조에만 사용했다. 다만 사용자가 이미 정답을 제공한 상태였으므로 공식 blind test로 분류하지 않는다.

### Ensemble 주요 후보

| 의미 | 순위 | seed | 생성 번호 | 적중 |
|---|---:|---:|---|---:|
| field score 1위 | 1 | 163185 | 4, 15, 17, 23, 27, 36 | 0 |
| Top-10 최고 | 9 | 163178 | 3, 15, 18, 31, 34, 37 | 2 |
| Top-100 공동 최고 | 34 | 163112 | 18, 21, 24, 26, 38, 43 | 3 |
| Top-100 공동 최고 | 35 | 163169 | 11, 13, 21, 23, 29, 38 | 3 |

모델별 Top-100의 최고 적중은 persistence 3, EWMA 3, analog 3, shift 3, XOR 2, ensemble 3, random 3이었다. 모든 모델에서 4+는 0건이었다. Shift 1순위 seed `225247`은 `3, 4, 18, 21, 28, 38`로 3개가 맞았지만, random Top-100 역시 3개까지 맞았으므로 차별적 성능으로 해석할 수 없다.

## 범위와 데이터

| 구분 | 회차 | 용도 | 상태 |
|---|---:|---|---|
| Historical warm-up | 820–851 | 852회 예측 전 최소 32개 field | 사용 |
| Historical | 852–1043 | 독립 시기 재현성 확인 | 평가 |
| Development | 1044–1235 | 주 walk-forward 평가 | 평가 |
| Locked | 660–851 | 승격 후 확인용 | **미사용·봉인** |
| Blind | 468–659 | 최종 확인용 | **미사용·봉인** |

- 원본: `lotto.xlsx`, 1–1235회, SHA-256 `7efe5e232c7d4ed347b0377726686adad93ebe0581f3d03257cf4e2f83836db4`
- Historical source: 820–1043회, 224개 회차, 224,000,000 seed 평가, qualifying 행 311,578개
- Development source: 1044–1235회, 192개 회차, 192,000,000 seed 평가, qualifying 행 266,849개
- 합계: 416,000,000 reverse seed 평가, qualifying 행 578,427개
- seed 공간: 회차마다 `[0, 1,000,000)`, 보너스 번호 제외
- 평가 회차: 각 cohort 192회, 합계 384회

## 알고리즘 명세

### 1. Answer-derived field

각 과거 회차의 4+/5+/6-hit seed에 질량을 부여한다.

```text
mass(seed) = hit_weight[hits] / (1 + positional_mae)
hit_weight[4]=1, hit_weight[5]=8, hit_weight[6]=64
```

각 채널에서 질량을 histogram으로 합산한 뒤 합이 1이 되도록 정규화한다.

### 2. 동결 좌표계 12개

- identity: 64, 256 bins
- Gray code: 64, 256 bins
- 20-bit reversal: 64, 256 bins
- 20-bit rotate-left 7: 64, 256 bins
- even/odd bit deinterleave: 64, 256 bins
- prime residue: modulo 257, modulo 509

### 3. 운반 연산자

- `persistence`: 직전 field 유지
- `EWMA`: 최근 16회, decay 0.9
- `analog`: 현재 field와 cosine 거리가 가까운 과거 5개 회차의 다음 field
- `shift`: 직전 두 field의 FFT circular shift를 다음 회차에 반복
- `XOR`: 직전 두 field의 최적 XOR mask를 다음 회차에 반복
- `ensemble`: EWMA·analog·shift·XOR의 동일 가중 평균
- `random`: 회차와 고정 namespace를 사용하는 matched deterministic baseline

각 seed는 12개 채널의 예측 밀도 평균으로 점수를 얻는다. 점수가 같으면 SplitMix 계열의 결정적 hash로 순서를 고정한다.

### 4. 고정 설정

- 최소 history: 32회
- budget: 10, 100, 1,000, 10,000
- random seed: 20260815
- null simulation: 10,000회
- 설정 fingerprint: `b0e78b49f0a7e0fa0f99a0b51e1a280868a6bc9043ad722cd6779f59b518960e`
- 결과 확인 전 core 구현 SHA-256: `c61f8e713fd51fb8c6ee8aa8a966e8baab5ece5e4d38892a4366ca779e2ac5bb`
- 출력·진행 로그·미래 후보 명령 추가 후 최종 구현 SHA-256: `1158f7421d50fb5ad9dd9928aabc9be6a467578812c66b34498acdb80dc94f46`

후자의 변경은 Top-10 seed CSV, 진행 로그, 정답 없는 후보 생성만 추가했다. 변경 전후 evaluation CSV와 summary JSON의 SHA-256이 각각 완전히 같아 모델 결과가 변하지 않았음을 확인했다.

## 검증 방법과 판정 규칙

모든 target 회차 `t`는 `t-1`까지의 field만 사용해 예측했다. `t`의 answer-derived field는 선택된 seed의 적중을 채점할 때만 조회했다. 각 회차의 4+/5+/6 qualifying seed 개수는 조금씩 다르므로 null은 그 개수를 고정한 hypergeometric sampling으로 구성했다.

승격 기준은 ensemble Top-1,000의 5+를 주 지표로 삼아 Development에서 lift ≥ 1.25와 `p≤0.01`, Historical에서 lift ≥ 1.20과 `p≤0.05`를 모두 요구하고, Top-10의 4+가 두 코호트에서 random보다 나쁘지 않을 것을 요구했다. 실제 결과는 첫 조건부터 모두 실패했다.

Locked와 Blind는 이 기준을 통과할 때만 연다. 이번에는 열지 않았다.

## 구현과 산출물

- `src/uriel_v2/seed_field.py`: field 구성, 연산자, scoring, walk-forward, null, 미래 후보 생성
- `src/uriel_v2/cli.py`: `seed-field`, `seed-field-predict`, 진행 로그와 CSV/JSON 저장
- `tests/test_seed_field.py`: 좌표 변환, 채널 범위, shift/XOR 복원, 결정성, CSV schema 검증
- `pyproject.toml`: NumPy 2.x 의존성
- `README.md`: 실행법과 결과 파일 설명

실행 중 `seed-field`는 16회 간격으로 진행률을 남기고, 마지막에 ensemble budget별 관측·lift·p-value를 출력한다. 실제 seed와 생성 번호는 `seed-field-top10-seeds.csv`, 다음 회차 후보는 `seed-field-candidates.csv`에서 확인할 수 있다. 대규모 reverse landscape 구축은 기존 8-worker process pool을 사용했고, Stage B의 100만 seed scoring은 프로세스 복제에 따른 메모리 증가를 피하기 위해 NumPy 벡터 연산으로 처리했다.

### 결과 파일 무결성

| 파일 | SHA-256 |
|---|---|
| Development evaluation CSV | `dde7b4441f8b1c093296bf6d69a2b95e0c206e5683701aa9ee7eb5993e0eeeb6` |
| Development summary JSON | `8b4594b082cd9471ffef9c7776317546ad43706bb48b8ca93b0544ee6b69d134` |
| Historical evaluation CSV | `a8c20a1b7352289848dac42b2fabb6028cd043e19929e91f7d3e91828a5b31b1` |
| Historical summary JSON | `3fc86a25306e6d809c120e3427af38aafe197a7f13aaf032e130a54819999ede` |
| Development Top-10 seed CSV | `585272cd7f417632520a8f0faee9646beb382c1c91c66f49a2495e2c84b84969` |
| Historical Top-10 seed CSV | `c9ad9645054f50dce1a83e0afba945d4bf4c69e0a4c289bebec4cbcc15fc70b2` |
| 1236 candidate CSV | `01e45cabf85af125f36ecac14e64046924012c003de276e7901968ee53c67134` |

## 재현 명령

```bash
PYTHONPATH=src python -m uriel_v2 seed-field \
  --data lotto.xlsx \
  --output outputs \
  --landscape outputs/reverse-dataset/20260813-211609-reverse-batch/reverse-hit-seeds.csv \
  --landscape outputs/reverse-dataset/20260813-132436-reverse-batch/reverse-hit-seeds.csv \
  --start-round 1044 --end-round 1235 --cohort Development

PYTHONPATH=src python -m uriel_v2 seed-field \
  --data lotto.xlsx \
  --output outputs \
  --landscape outputs/reverse-dataset/20260813-211609-reverse-batch/reverse-hit-seeds.csv \
  --landscape outputs/reverse-dataset/20260813-132436-reverse-batch/reverse-hit-seeds.csv \
  --start-round 852 --end-round 1043 --cohort Historical

PYTHONPATH=src python -m uriel_v2 seed-field-predict \
  --data lotto.xlsx \
  --output outputs \
  --landscape outputs/reverse-dataset/20260813-211609-reverse-batch/reverse-hit-seeds.csv \
  --landscape outputs/reverse-dataset/20260813-132436-reverse-batch/reverse-hit-seeds.csv \
  --round 1236 --top-k 100

PYTHONPATH=src:. python -m unittest discover -s tests -v
```

최종 테스트는 26개 모두 통과했다.

## 한계와 강건성

- 과거 field는 정답에서 역산된 4+ seed로 만들었다. 예측 대상 회차의 정답은 scoring 전까지 사용하지 않았지만, 분석 전체는 answer-derived 구조에 의존한다.
- 좌표계와 연산자는 결과 확인 전에 고정했지만, 가능한 비정상 변환의 극히 일부만 다룬다.
- 여러 모델·budget·threshold를 함께 보면 nominal p-value가 작은 이상점이 우연히 생긴다. 따라서 primary 지표와 독립 시기 재현을 우선했다.
- Monte Carlo p-value의 최소 해상도는 약 0.0001이며, 10,000회 표본 오차가 있다.
- 1236회 대조는 알고리즘 입력상 미래 누출이 없지만, 사용자가 정답을 이미 알려준 뒤 수행했으므로 공식 blind evidence가 아니다.
- 로또 추첨이 독립 무작위라는 기본 가정 아래, 과거 번호나 seed landscape만으로 지속적인 예측 우위를 얻을 근거는 현재 없다.

## 다음 단계

1. 이 Seed Field Transport ensemble은 종료한다. 동일 좌표계의 가중치 튜닝으로 결과를 구제하지 않는다.
2. persistence Top-100 exact-6와 shift Top-10은 **새 가설의 씨앗**으로만 보존한다. 다시 검정한다면 모델·budget·지표를 하나로 제한한 별도 사전 계획을 먼저 작성한다.
3. Locked/Blind는 그대로 보존한다. 새로운 단일 가설이 Development에서 미리 정한 기준을 통과하기 전에는 열지 않는다.
4. 다음 비정상 방향은 “seed 값의 이동”보다 **생성기 내부 상태의 부분 제약**을 역추론하는 구조적 decoder가 더 적합하다. 단, SplitMix64의 avalanche 특성 때문에 먼저 축소 장난감 공간에서 복구 가능성을 증명한 뒤 실제 20-bit seed 공간으로 확장해야 한다.

## 추가로 답해야 할 질문

- 1148회 persistence rank 66 exact-6는 동일 규칙의 새 데이터에서도 다시 나타나는가?
- circular shift의 Historical Top-10 약신호는 특정 시기 regime에 국한됐는가, 아니면 좌표계 하나가 대부분을 설명하는가?
- seed의 출력 번호가 아니라 SplitMix64 내부 상태의 비트 제약을 사용하면 answer-derived 역산 집합의 정보 손실을 줄일 수 있는가?
- 완전 독립 새 회차가 충분히 쌓이기 전까지 어떤 최소 표본과 효과 크기를 다음 승격 기준으로 둘 것인가?

