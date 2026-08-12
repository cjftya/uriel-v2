# Uriel v2

한국 로또 6/45 데이터를 이용해 **재현 가능한 시드 생성 방법**을 단순하게 비교하는 Python 실험 프로젝트입니다. UI 없이 CLI, 로그, CSV/JSON 결과에 집중합니다.

현재 v0.1의 범위는 다음과 같습니다.

- `lotto.xlsx` 구조 자동 인식 및 데이터 검증
- Python 버전과 무관하게 같은 결과를 내는 SplitMix64 번호 생성기
- 미래 당첨번호를 사용하지 않는 3가지 시드 전략
- Process worker 기반 walk-forward 평가
- 실제 정답 번호로 시드 범위를 탐색하는 역산 진단
- 콘솔 로그와 실행별 `outputs/` 결과 보존

> 역산 탐색은 이미 알려진 정답을 사용합니다. 시드 공간의 구조와 근접도를 살피는 진단 실험이지, 그 자체가 미래 회차 예측은 아닙니다.

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
