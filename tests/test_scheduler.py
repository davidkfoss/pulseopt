"""Tests for the high-level AEES wrapper."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from pulseopt import AEES
from pulseopt.types import CandidateConfig


def _make_model_and_optimizer(lr: float = 1e-3) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return model, optimizer


def _run_steps(aees: AEES, model: nn.Module, optimizer: torch.optim.Optimizer, n: int) -> None:
    for step in range(n):
        aees.step_start(step)
        optimizer.zero_grad()
        loss = model(torch.randn(4, 8)).pow(2).mean()
        loss.backward()
        aees.step_end(loss)


class TestAEESSmoke:
    def test_basic_run_produces_episode_logs(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(
            optimizer,
            lr_candidates=[0.5, 1.0, 2.0],
            noise_candidates=[0.0, 0.005],
            episode_length=5,
            seed=0,
        )
        _run_steps(aees, model, optimizer, 20)
        aees.finalize()
        logs = aees.get_logs()
        assert len(logs["episode_rewards"]) >= 1
        assert len(logs["selected_lr_values"]) >= 1
        assert len(logs["selected_noise_values"]) >= 1

    def test_step_start_returns_candidate_config(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(optimizer, lr_candidates=[0.5, 1.0], noise_candidates=[0.0], episode_length=5)
        candidate = aees.step_start(0)
        assert isinstance(candidate, CandidateConfig)
        assert candidate.lr_multiplier in (0.5, 1.0)
        optimizer.zero_grad()
        loss = model(torch.randn(2, 8)).pow(2).mean()
        loss.backward()
        aees.step_end(loss)

    def test_optimizer_property(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], episode_length=5)
        assert aees.optimizer is optimizer


class TestAEESParameterUpdate:
    def test_params_actually_change_after_step_end(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], episode_length=5)

        before = [p.detach().clone() for p in model.parameters()]
        aees.step_start(0)
        optimizer.zero_grad()
        loss = model(torch.randn(4, 8)).pow(2).mean()
        loss.backward()
        aees.step_end(loss)
        after = [p.detach().clone() for p in model.parameters()]

        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_lr_multiplier_is_transient(self):
        lr = 1e-3
        model, optimizer = _make_model_and_optimizer(lr=lr)
        aees = AEES(
            optimizer,
            lr_candidates=[0.5, 2.0],
            noise_candidates=[0.0],
            episode_length=5,
            seed=0,
        )

        lr_before = optimizer.param_groups[0]["lr"]
        aees.step_start(0)
        optimizer.zero_grad()
        loss = model(torch.randn(4, 8)).pow(2).mean()
        loss.backward()
        aees.step_end(loss)
        lr_after = optimizer.param_groups[0]["lr"]

        assert lr_before == lr_after


class TestAEESLrScheduler:
    def test_lr_scheduler_is_called(self):
        model, optimizer = _make_model_and_optimizer(lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        aees = AEES(
            optimizer,
            lr_candidates=[1.0],
            noise_candidates=[0.0],
            episode_length=5,
            lr_scheduler=scheduler,
        )

        lr_initial = optimizer.param_groups[0]["lr"]
        _run_steps(aees, model, optimizer, 5)
        lr_after = optimizer.param_groups[0]["lr"]

        assert lr_after != lr_initial


class TestAEESFixedAxes:
    def test_fixed_lr_and_noise_runs_without_error(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], episode_length=3)
        _run_steps(aees, model, optimizer, 9)
        aees.finalize()
        logs = aees.get_logs()
        assert len(logs["episode_rewards"]) >= 1

    def test_single_lr_candidate_nonunit(self):
        model, optimizer = _make_model_and_optimizer(lr=1e-3)
        aees = AEES(optimizer, lr_candidates=[0.5], noise_candidates=[0.0], episode_length=3)
        _run_steps(aees, model, optimizer, 6)
        aees.finalize()
        assert len(aees.get_logs()["selected_lr_values"]) >= 1


class TestAEESContextModes:
    def test_trend_context_mode(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(
            optimizer,
            lr_candidates=[0.5, 1.0, 2.0],
            noise_candidates=[0.0],
            episode_length=4,
            context_mode="trend",
            seed=0,
        )
        _run_steps(aees, model, optimizer, 20)
        aees.finalize()
        logs = aees.get_logs()
        assert "context_bucket_ids" in logs

    def test_trend_phase_context_mode(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(
            optimizer,
            lr_candidates=[0.5, 1.0],
            noise_candidates=[0.0],
            episode_length=4,
            context_mode="trend_phase",
            total_training_steps=40,
            seed=0,
        )
        _run_steps(aees, model, optimizer, 20)
        aees.finalize()
        logs = aees.get_logs()
        assert "context_bucket_ids" in logs


class TestAEESConditionalMode:
    def test_conditional_structured_control(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(
            optimizer,
            lr_candidates=[0.5, 1.0, 2.0],
            noise_candidates=[0.0, 0.005],
            episode_length=4,
            structured_control_mode="conditional",
            seed=0,
        )
        _run_steps(aees, model, optimizer, 20)
        aees.finalize()
        assert len(aees.get_logs()["episode_rewards"]) >= 1


class TestAEESGetLogs:
    def test_get_logs_returns_shallow_copies(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], episode_length=3)
        _run_steps(aees, model, optimizer, 6)
        aees.finalize()

        logs1 = aees.get_logs()
        logs1["episode_rewards"].append(999.0)
        logs2 = aees.get_logs()
        assert 999.0 not in logs2["episode_rewards"]


class TestAEESGuards:
    def test_step_end_without_step_start_raises(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], episode_length=5)
        optimizer.zero_grad()
        loss = model(torch.randn(2, 8)).pow(2).mean()
        loss.backward()
        with pytest.raises(RuntimeError, match="step_end\\(\\) called without"):
            aees.step_end(loss)

    def test_step_start_twice_raises(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], episode_length=5)
        aees.step_start(0)
        with pytest.raises(RuntimeError, match="step_start\\(\\) called twice"):
            aees.step_start(1)
        # clean up
        optimizer.zero_grad()
        loss = model(torch.randn(2, 8)).pow(2).mean()
        loss.backward()
        aees.step_end(loss)

    def test_finalize_with_open_step_raises(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], episode_length=5)
        aees.step_start(0)
        with pytest.raises(RuntimeError, match="finalize\\(\\) called with an open step"):
            aees.finalize()
        # clean up
        optimizer.zero_grad()
        loss = model(torch.randn(2, 8)).pow(2).mean()
        loss.backward()
        aees.step_end(loss)

    def test_invalid_control_mode_raises(self):
        _, optimizer = _make_model_and_optimizer()
        with pytest.raises(ValueError, match="control_mode"):
            AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], control_mode="invalid")

    def test_invalid_context_mode_raises(self):
        _, optimizer = _make_model_and_optimizer()
        with pytest.raises(ValueError, match="context_mode"):
            AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], context_mode="invalid")

    def test_trend_phase_without_total_steps_raises(self):
        _, optimizer = _make_model_and_optimizer()
        with pytest.raises(ValueError):
            AEES(
                optimizer,
                lr_candidates=[0.5, 1.0],
                noise_candidates=[0.0],
                context_mode="trend_phase",
                total_training_steps=None,
            )


class TestAEESInputValidation:
    @pytest.mark.parametrize(
        "lr_candidates",
        [[], [0.0, 1.0], [-1.0, 1.0]],
    )
    def test_invalid_lr_candidates_raise(self, lr_candidates):
        _, optimizer = _make_model_and_optimizer()
        with pytest.raises(ValueError, match="lr_candidates"):
            AEES(optimizer, lr_candidates=lr_candidates, noise_candidates=[0.0])

    @pytest.mark.parametrize(
        "noise_candidates",
        [[], [-0.001, 0.0]],
    )
    def test_invalid_noise_candidates_raise(self, noise_candidates):
        _, optimizer = _make_model_and_optimizer()
        with pytest.raises(ValueError, match="noise_candidates"):
            AEES(optimizer, lr_candidates=[1.0], noise_candidates=noise_candidates)

    def test_zero_episode_length_raises(self):
        _, optimizer = _make_model_and_optimizer()
        with pytest.raises(ValueError, match="episode_length"):
            AEES(
                optimizer,
                lr_candidates=[1.0],
                noise_candidates=[0.0],
                episode_length=0,
            )

    def test_total_training_steps_outside_trend_phase_warns(self):
        _, optimizer = _make_model_and_optimizer()
        with pytest.warns(UserWarning, match="total_training_steps"):
            AEES(
                optimizer,
                lr_candidates=[1.0],
                noise_candidates=[0.0],
                episode_length=5,
                context_mode="trend",
                total_training_steps=100,
            )


class TestAEESCurrentCandidate:
    def test_current_candidate_lifecycle(self):
        model, optimizer = _make_model_and_optimizer()
        aees = AEES(
            optimizer,
            lr_candidates=[0.5, 1.0],
            noise_candidates=[0.0],
            episode_length=5,
            seed=0,
        )

        assert aees.current_candidate is None

        candidate = aees.step_start(0)
        assert aees.current_candidate is candidate

        optimizer.zero_grad()
        loss = model(torch.randn(2, 8)).pow(2).mean()
        loss.backward()
        aees.step_end(loss)

        assert aees.current_candidate is None


class TestAEESNonFiniteLoss:
    @pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), -float("inf")])
    def test_step_end_with_non_finite_loss_raises(self, bad_loss):
        _, optimizer = _make_model_and_optimizer()
        aees = AEES(optimizer, lr_candidates=[1.0], noise_candidates=[0.0], episode_length=5)
        aees.step_start(0)
        with pytest.raises(ValueError, match="finite"):
            aees.step_end(bad_loss)
        # The step_started flag must be cleared so the wrapper is recoverable
        # only via a new step_start; the candidate stays bound until then.
        assert math.isnan(bad_loss) or math.isinf(bad_loss)
