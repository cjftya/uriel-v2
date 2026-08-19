from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


SCHEMA_VERSION = "1.0.0"
DEFAULT_CHECKPOINTS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for IDs and extension fields."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _optional_float(value: float | int | None) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    problem_family: str
    domain: str
    problem_seed: int
    dimension: int | None = None
    size: int | None = None
    density: float | None = None
    sparsity: float | None = None
    noise: float | None = None
    entropy: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    autocorrelation: float | None = None
    condition_number: float | None = None
    spectral_decay: float | None = None
    multimodality: float | None = None
    ruggedness: float | None = None
    effective_dimension: float | None = None
    extension: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "problem_id": self.problem_id,
            "problem_family": self.problem_family,
            "domain": self.domain,
            "problem_seed": int(self.problem_seed),
            "dimension": self.dimension,
            "size": self.size,
            "density": _optional_float(self.density),
            "sparsity": _optional_float(self.sparsity),
            "noise": _optional_float(self.noise),
            "entropy": _optional_float(self.entropy),
            "skewness": _optional_float(self.skewness),
            "kurtosis": _optional_float(self.kurtosis),
            "autocorrelation": _optional_float(self.autocorrelation),
            "condition_number": _optional_float(self.condition_number),
            "spectral_decay": _optional_float(self.spectral_decay),
            "multimodality": _optional_float(self.multimodality),
            "ruggedness": _optional_float(self.ruggedness),
            "effective_dimension": _optional_float(self.effective_dimension),
            "extension_json": canonical_json(dict(self.extension)),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ProblemSpec":
        def optional_value(name: str) -> Any:
            value = record.get(name)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            return value

        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"problem schema mismatch: {record.get('schema_version')}")
        extension = record.get("extension")
        if extension is None:
            extension = json.loads(str(record.get("extension_json", "{}")))
        return cls(
            problem_id=str(record["problem_id"]),
            problem_family=str(record["problem_family"]),
            domain=str(record["domain"]),
            problem_seed=int(record["problem_seed"]),
            dimension=None if optional_value("dimension") is None else int(optional_value("dimension")),
            size=None if optional_value("size") is None else int(optional_value("size")),
            density=optional_value("density"),
            sparsity=optional_value("sparsity"),
            noise=optional_value("noise"),
            entropy=optional_value("entropy"),
            skewness=optional_value("skewness"),
            kurtosis=optional_value("kurtosis"),
            autocorrelation=optional_value("autocorrelation"),
            condition_number=optional_value("condition_number"),
            spectral_decay=optional_value("spectral_decay"),
            multimodality=optional_value("multimodality"),
            ruggedness=optional_value("ruggedness"),
            effective_dimension=optional_value("effective_dimension"),
            extension=extension,
        )


@dataclass(frozen=True)
class AlgorithmSpec:
    algorithm: str
    algorithm_family: str
    random_mechanism: str
    version: str = "1"
    configuration: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BudgetSpec:
    budget_type: str
    total: int
    checkpoints: tuple[float, ...] = DEFAULT_CHECKPOINTS

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("budget total must be positive")
        if not self.checkpoints:
            raise ValueError("at least one budget checkpoint is required")
        if any(fraction <= 0.0 or fraction > 1.0 for fraction in self.checkpoints):
            raise ValueError("budget checkpoints must be in (0, 1]")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("budget checkpoints must be unique and sorted")
        if self.checkpoints[-1] != 1.0:
            raise ValueError("budget checkpoints must include 1.0")


@dataclass(frozen=True)
class JobSpec:
    problem: ProblemSpec
    algorithm: AlgorithmSpec
    seed: int
    budget: BudgetSpec
    rng_algorithm: str = "PCG64"
    rng_version: str = "NumPy"

    @property
    def run_id(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "problem": self.problem.to_record(),
            "algorithm": asdict(self.algorithm),
            "seed": int(self.seed),
            "budget": asdict(self.budget),
            "rng_algorithm": self.rng_algorithm,
            "rng_version": self.rng_version,
        }
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        return digest[:24]


@dataclass(frozen=True)
class TracePoint:
    run_id: str
    step: int
    budget_fraction: float
    elapsed_time: float
    objective: float
    best_so_far: float
    improvement: float
    improvement_rate: float
    variance: float
    entropy: float
    diversity: float
    distance_to_best: float
    distance_to_target: float
    failure_signal: float
    extension: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["schema_version"] = SCHEMA_VERSION
        record["extension_json"] = canonical_json(record.pop("extension"))
        return record


@dataclass(frozen=True)
class RunResult:
    run_id: str
    problem_id: str
    problem_family: str
    domain: str
    algorithm: str
    algorithm_family: str
    random_mechanism: str
    algorithm_version: str
    seed: int
    rng_algorithm: str
    rng_version: str
    budget_type: str
    budget: int
    status: str
    quality_final: float | None
    quality_best: float | None
    runtime: float
    steps: int
    success: bool
    failure: bool
    timeout: bool
    target_reached: bool
    first_passage_time: int | None
    t50: int | None
    t75: int | None
    t90: int | None
    t95: int | None
    t99: int | None
    mean_quality: float | None
    variance_quality: float | None
    best_so_far: float | None
    improvement_rate: float | None
    stagnation: int | None
    failure_type: str | None = None
    failure_time: float | None = None
    algorithm_config: Mapping[str, Any] = field(default_factory=dict)
    extension: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["schema_version"] = SCHEMA_VERSION
        record["algorithm_config_json"] = canonical_json(record.pop("algorithm_config"))
        record["extension_json"] = canonical_json(record.pop("extension"))
        return record


@dataclass(frozen=True)
class ExperimentBundle:
    result: RunResult
    traces: tuple[TracePoint, ...]

    def to_checkpoint_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "result": self.result.to_record(),
            "traces": [trace.to_record() for trace in self.traces],
        }
