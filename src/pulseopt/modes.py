"""Helpers for parsing and assembling LR/noise candidate configs."""

from __future__ import annotations

from pulseopt.types import CandidateConfig


def parse_lr_candidates(lr_candidates_str: str) -> list[float]:
    """Parse a comma-separated list of LR multipliers."""

    return _parse_float_candidates(
        values_str=lr_candidates_str,
        field_name="lr_candidates",
        positive_only=True,
    )


def parse_noise_candidates(noise_candidates_str: str) -> list[float]:
    """Parse a comma-separated list of gradient-noise values."""

    return _parse_float_candidates(
        values_str=noise_candidates_str,
        field_name="noise_candidates",
        positive_only=False,
    )


def build_candidate_configs_from_axes(
    lr_candidates: list[float],
    noise_candidates: list[float],
) -> list[CandidateConfig]:
    """Build candidate configs from the Cartesian product of LR and noise values."""

    if not lr_candidates:
        raise ValueError("lr_candidates must contain at least one value.")
    if not noise_candidates:
        raise ValueError("noise_candidates must contain at least one value.")

    candidate_configs: list[CandidateConfig] = []
    for lr_multiplier in lr_candidates:
        for noise_std in noise_candidates:
            candidate_configs.append(
                CandidateConfig(
                    name=build_generated_candidate_name(lr_multiplier, noise_std),
                    lr_multiplier=lr_multiplier,
                    noise_std=noise_std,
                )
            )
    return candidate_configs


def build_generated_candidate_name(lr_multiplier: float, noise_std: float) -> str:
    """Build a stable readable candidate name for an LR/noise pair."""

    return f"lr{_format_float_token(lr_multiplier)}_n{_format_float_token(noise_std)}"


def _parse_float_candidates(
    values_str: str,
    field_name: str,
    positive_only: bool,
) -> list[float]:
    raw_values = [value.strip() for value in values_str.split(",") if value.strip()]
    if not raw_values:
        raise ValueError(f"{field_name} must contain at least one value.")

    parsed_values: list[float] = []
    seen_values: set[float] = set()
    for raw_value in raw_values:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid value '{raw_value}' in {field_name}.") from exc
        if positive_only and value <= 0.0:
            raise ValueError(f"{field_name} values must be positive.")
        if not positive_only and value < 0.0:
            raise ValueError(f"{field_name} values must be non-negative.")
        if value in seen_values:
            raise ValueError(f"{field_name} contains a duplicate value: {value}.")
        seen_values.add(value)
        parsed_values.append(value)
    return parsed_values


def _format_float_token(value: float) -> str:
    text = f"{value:.12g}"
    text = text.replace("-", "m").replace(".", "p")
    return text
