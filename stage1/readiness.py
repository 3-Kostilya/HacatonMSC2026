"""Explicit completion gates for Stage 1 and the following forecast stage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReadinessEvidence:
    types_covered: int
    groups_covered: int
    synthetic_scenarios_covered: int
    real_examples_reviewed: int
    causal_checks_passed: bool
    reproducible_run_available: bool
    temporal_protocol_frozen: bool
    full_history_candidate_catalog: bool
    candidate_diversity_sufficient: bool
    candidates_traceable: bool
    external_validation_available: bool


@dataclass(frozen=True, slots=True)
class ReadinessDecision:
    stage_complete: bool
    research_forecast_ready: bool
    operational_claim_ready: bool
    completed_checks: tuple[str, ...]
    blockers: tuple[str, ...]


def assess_readiness(evidence: ReadinessEvidence) -> ReadinessDecision:
    completed: list[str] = []
    blockers: list[str] = []
    method_checks = {
        "all_19_types_covered": evidence.types_covered == 19,
        "all_7_groups_covered": evidence.groups_covered == 7,
        "synthetic_protocol_covered": evidence.synthetic_scenarios_covered >= 9,
        "real_examples_reviewed": evidence.real_examples_reviewed >= 30,
        "causal_checks_passed": evidence.causal_checks_passed,
        "reproducible_run_available": evidence.reproducible_run_available,
        "temporal_protocol_frozen": evidence.temporal_protocol_frozen,
        "candidates_traceable": evidence.candidates_traceable,
    }
    for name, passed in method_checks.items():
        (completed if passed else blockers).append(name if passed else f"missing:{name}")
    stage_complete = all(method_checks.values())

    forecast_checks = {
        "full_history_candidate_catalog": evidence.full_history_candidate_catalog,
        "candidate_diversity_sufficient": evidence.candidate_diversity_sufficient,
    }
    for name, passed in forecast_checks.items():
        (completed if passed else blockers).append(name if passed else f"missing:{name}")
    research_ready = stage_complete and all(forecast_checks.values())

    if evidence.external_validation_available:
        completed.append("external_validation_available")
    else:
        blockers.append("missing:external_validation_available")
    return ReadinessDecision(
        stage_complete=stage_complete,
        research_forecast_ready=research_ready,
        operational_claim_ready=research_ready and evidence.external_validation_available,
        completed_checks=tuple(completed),
        blockers=tuple(blockers),
    )
