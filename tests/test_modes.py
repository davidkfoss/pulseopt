"""Tests for candidate-config parsing and assembly."""

import pytest

from pulseopt.modes import (
    build_candidate_configs_from_axes,
    parse_lr_candidates,
    parse_noise_candidates,
)


def test_build_candidate_configs_from_lr_and_noise_axes() -> None:
    """Structured candidates should be the LR/noise Cartesian product."""

    candidates = build_candidate_configs_from_axes(
        lr_candidates=[1.0, 1.2],
        noise_candidates=[0.0, 0.01],
    )

    assert [(candidate.lr_multiplier, candidate.noise_std) for candidate in candidates] == [
        (1.0, 0.0),
        (1.0, 0.01),
        (1.2, 0.0),
        (1.2, 0.01),
    ]


def test_build_candidate_configs_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="lr_candidates"):
        build_candidate_configs_from_axes(lr_candidates=[], noise_candidates=[0.0])
    with pytest.raises(ValueError, match="noise_candidates"):
        build_candidate_configs_from_axes(lr_candidates=[1.0], noise_candidates=[])


def test_parse_lr_candidates_accepts_positive_values() -> None:
    assert parse_lr_candidates("0.5, 1.0, 2.0") == [0.5, 1.0, 2.0]


@pytest.mark.parametrize("bad", ["", "0,1", "-1,1", "1,1"])
def test_parse_lr_candidates_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError, match="lr_candidates"):
        parse_lr_candidates(bad)


def test_parse_noise_candidates_accepts_zero() -> None:
    assert parse_noise_candidates("0.0,0.005") == [0.0, 0.005]


@pytest.mark.parametrize("bad", ["", "-0.001,0", "0,0"])
def test_parse_noise_candidates_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError, match="noise_candidates"):
        parse_noise_candidates(bad)
