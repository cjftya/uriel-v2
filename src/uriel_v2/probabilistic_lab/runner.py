from __future__ import annotations

import traceback
from time import perf_counter

import numpy as np

from uriel_v2.probabilistic_lab.algorithms import default_registry
from uriel_v2.probabilistic_lab.schema import ExperimentBundle, JobSpec, RunResult


def execute_job(job: JobSpec) -> ExperimentBundle:
    """Execute one job with an isolated RNG and convert exceptions to typed failures."""

    started_at = perf_counter()
    try:
        bit_generator = getattr(np.random, job.rng_algorithm)
    except AttributeError as exc:
        return _failure_bundle(job, started_at, "FAIL_RNG", exc)
    try:
        rng = np.random.Generator(bit_generator(job.seed))
        algorithm = default_registry().resolve(job.algorithm.algorithm, job.problem.domain)
        bundle = algorithm(job, rng)
        if not bundle.traces:
            raise ValueError("successful algorithm returned no trace")
        return bundle
    except (FloatingPointError, OverflowError) as exc:
        return _failure_bundle(job, started_at, "FAIL_NUMERIC", exc)
    except MemoryError as exc:
        return _failure_bundle(job, started_at, "FAIL_MEMORY", exc)
    except Exception as exc:  # noqa: BLE001 - failures are research data
        return _failure_bundle(job, started_at, "FAIL_EXECUTION", exc)


def _failure_bundle(job: JobSpec, started_at: float, failure_type: str, exc: BaseException) -> ExperimentBundle:
    runtime = perf_counter() - started_at
    result = RunResult(
        run_id=job.run_id,
        problem_id=job.problem.problem_id,
        problem_family=job.problem.problem_family,
        domain=job.problem.domain,
        algorithm=job.algorithm.algorithm,
        algorithm_family=job.algorithm.algorithm_family,
        random_mechanism=job.algorithm.random_mechanism,
        algorithm_version=job.algorithm.version,
        seed=job.seed,
        rng_algorithm=job.rng_algorithm,
        rng_version=job.rng_version,
        budget_type=job.budget.budget_type,
        budget=job.budget.total,
        status=failure_type,
        quality_final=None,
        quality_best=None,
        runtime=runtime,
        steps=0,
        success=False,
        failure=True,
        timeout=failure_type == "FAIL_TIMEOUT",
        target_reached=False,
        first_passage_time=None,
        t50=None,
        t75=None,
        t90=None,
        t95=None,
        t99=None,
        mean_quality=None,
        variance_quality=None,
        best_so_far=None,
        improvement_rate=None,
        stagnation=None,
        failure_type=failure_type,
        failure_time=runtime,
        algorithm_config=job.algorithm.configuration,
        extension={
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8_000:],
        },
    )
    return ExperimentBundle(result=result, traces=())
