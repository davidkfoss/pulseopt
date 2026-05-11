"""Tests for the AdamW mode wrapper."""

from __future__ import annotations

import pytest
import torch

from pulseopt.optimizer import AdaptiveModeAdamW
from pulseopt.types import CandidateConfig

pytestmark = pytest.mark.filterwarnings("ignore:AdaptiveModeAdamW is deprecated:DeprecationWarning")


def build_model(weight: float = 1.0) -> torch.nn.Linear:
    """Create a tiny deterministic linear model."""

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight)
    return model


def compute_loss(model: torch.nn.Module) -> torch.Tensor:
    """Return a simple quadratic-style MSE loss."""

    inputs = torch.tensor([[1.0]])
    targets = torch.tensor([[0.0]])
    predictions = model(inputs)
    return torch.nn.functional.mse_loss(predictions, targets)


def test_wrapper_can_optimize_tiny_model() -> None:
    """The wrapper should run a few AdamW steps without error."""

    model = build_model(weight=1.0)
    optimizer = AdaptiveModeAdamW(model.parameters(), lr=0.1)

    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        loss = compute_loss(model)
        loss.backward()
        losses.append(float(loss.item()))
        optimizer.step()

    assert losses[-1] < losses[0]
    assert optimizer.compute_update_norm() is not None


def test_set_mode_changes_effective_learning_behavior() -> None:
    """A larger mode multiplier should produce a larger one-step update."""

    base_model = build_model(weight=1.0)
    aggressive_model = build_model(weight=1.0)

    base_optimizer = AdaptiveModeAdamW(base_model.parameters(), lr=0.1)
    aggressive_optimizer = AdaptiveModeAdamW(aggressive_model.parameters(), lr=0.1)
    aggressive_optimizer.set_mode(
        CandidateConfig(name="aggressive_test", lr_multiplier=2.0, noise_std=0.0)
    )

    initial_weight = 1.0
    for model, optimizer in [
        (base_model, base_optimizer),
        (aggressive_model, aggressive_optimizer),
    ]:
        optimizer.zero_grad()
        loss = compute_loss(model)
        loss.backward()
        optimizer.step()

    base_change = abs(float(base_model.weight.item()) - initial_weight)
    aggressive_change = abs(float(aggressive_model.weight.item()) - initial_weight)
    assert aggressive_change > base_change


def test_temporary_lr_scaling_is_restored_after_step() -> None:
    """Scaled LRs should not permanently overwrite the configured base LRs."""

    model = build_model(weight=1.0)
    optimizer = AdaptiveModeAdamW(model.parameters(), lr=0.1)
    optimizer.set_mode(CandidateConfig(name="scaled", lr_multiplier=1.5))

    initial_group_lrs = [group["lr"] for group in optimizer.param_groups]

    optimizer.zero_grad()
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()

    assert initial_group_lrs == [0.1]
    assert [group["lr"] for group in optimizer.param_groups] == [0.1]
    assert optimizer.state_dict()["base_lrs"] == [0.1]


def test_cosine_scheduler_updates_scheduled_base_lr_without_being_reset() -> None:
    """Scheduler-updated base LR should remain intact across AEES-scaled steps."""

    model = build_model(weight=1.0)
    optimizer = AdaptiveModeAdamW(model.parameters(), lr=0.1)
    optimizer.set_mode(CandidateConfig(name="scaled", lr_multiplier=2.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    observed_base_lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(2):
        optimizer.zero_grad()
        loss = compute_loss(model)
        loss.backward()
        optimizer.step()
        observed_base_lrs.append(float(optimizer.param_groups[0]["lr"]))
        scheduler.step()
        observed_base_lrs.append(float(optimizer.param_groups[0]["lr"]))

    assert observed_base_lrs[0] == 0.1
    assert observed_base_lrs[1] == 0.1
    assert observed_base_lrs[2] < observed_base_lrs[0]
    assert observed_base_lrs[3] == observed_base_lrs[2]
    assert observed_base_lrs[4] < observed_base_lrs[2]


def test_state_dict_round_trip_preserves_optimizer_and_mode_state() -> None:
    """Loading wrapper state should preserve later optimization behavior."""

    mode = CandidateConfig(name="aggressive", lr_multiplier=1.5)

    model_a = build_model(weight=1.0)
    optimizer_a = AdaptiveModeAdamW(model_a.parameters(), lr=0.05)
    optimizer_a.set_mode(mode)

    optimizer_a.zero_grad()
    loss_a = compute_loss(model_a)
    loss_a.backward()
    optimizer_a.step()

    saved_model_state = model_a.state_dict()
    saved_optimizer_state = optimizer_a.state_dict()

    model_b = build_model(weight=0.0)
    model_b.load_state_dict(saved_model_state)
    optimizer_b = AdaptiveModeAdamW(model_b.parameters(), lr=0.05)
    optimizer_b.load_state_dict(saved_optimizer_state)

    state_b = optimizer_b.state_dict()
    assert state_b["mode"] == saved_optimizer_state["mode"]
    assert state_b["base_lrs"] == saved_optimizer_state["base_lrs"]

    for model, optimizer in [(model_a, optimizer_a), (model_b, optimizer_b)]:
        optimizer.zero_grad()
        loss = compute_loss(model)
        loss.backward()
        optimizer.step()

    assert torch.allclose(model_a.weight, model_b.weight, atol=1e-4, rtol=1e-4)


def test_zero_grad_delegates_to_wrapped_optimizer() -> None:
    """zero_grad should clear accumulated gradients on the wrapped optimizer."""

    model = build_model(weight=1.0)
    optimizer = AdaptiveModeAdamW(model.parameters(), lr=0.1)

    loss = compute_loss(model)
    loss.backward()
    assert model.weight.grad is not None

    optimizer.zero_grad(set_to_none=True)
    assert model.weight.grad is None


def test_noise_mode_is_reproducible_with_fixed_seed() -> None:
    """Gradient noise should be reproducible for identical seeds."""

    mode = CandidateConfig(name="noisy_test", noise_std=0.05)

    model_a = build_model(weight=1.0)
    model_b = build_model(weight=1.0)

    optimizer_a = AdaptiveModeAdamW(model_a.parameters(), lr=0.1, mode=mode, noise_seed=1234)
    optimizer_b = AdaptiveModeAdamW(model_b.parameters(), lr=0.1, mode=mode, noise_seed=1234)

    for model, optimizer in [(model_a, optimizer_a), (model_b, optimizer_b)]:
        optimizer.zero_grad()
        loss = compute_loss(model)
        loss.backward()
        optimizer.step()

    assert torch.allclose(model_a.weight, model_b.weight)


def test_structured_optimizer_keeps_weight_decay_fixed() -> None:
    """Structured AEES should not adapt weight decay through the mode."""

    model = build_model(weight=1.0)
    optimizer = AdaptiveModeAdamW(model.parameters(), lr=0.1, weight_decay=0.01)
    optimizer.set_mode(CandidateConfig(name="scaled", lr_multiplier=1.5, noise_std=0.0))

    optimizer.zero_grad()
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()

    state = optimizer.state_dict()
    assert state["base_weight_decays"] == [0.01]
    assert "weight_decay_multiplier" not in state["mode"]
