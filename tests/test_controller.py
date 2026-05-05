"""Tests for controller implementations."""

from pulseopt.controller import (
    BucketedContextualController,
    DiscountedUCBController,
    RandomController,
)


def test_discounted_ucb_warmup_covers_all_arms_before_ucb() -> None:
    """Warmup should ensure each arm is selected the requested number of times."""

    controller = DiscountedUCBController(n_arms=3, warmup_rounds=2, random_seed=7)

    selected = []
    for _ in range(6):
        mode_id = controller.select_mode()
        selected.append(mode_id)
        controller.update(mode_id, reward=0.0)

    assert sorted(selected) == [0, 0, 1, 1, 2, 2]
    assert controller.get_state()["warmup_counts"] == [2, 2, 2]


def test_discounted_ucb_update_changes_state_sensibly() -> None:
    """Updates should discount old statistics and add the new observation."""

    controller = DiscountedUCBController(n_arms=2, discount=0.5, warmup_rounds=0)

    controller.update(mode_id=1, reward=2.5)
    state_after_first = controller.get_state()
    assert state_after_first["counts"] == [0.0, 1.0]
    assert state_after_first["reward_sums"] == [0.0, 2.5]
    assert state_after_first["mean_rewards"] == [0.0, 2.5]

    controller.update(mode_id=0, reward=1.0)
    state_after_second = controller.get_state()
    counts = state_after_second["counts"]
    reward_sums = state_after_second["reward_sums"]

    assert counts == [1.0, 0.5]
    assert reward_sums == [1.0, 1.25]
    assert state_after_second["total_updates"] == 2


def test_discounted_ucb_counts_remain_bounded_under_discounting() -> None:
    """Discounted counts should asymptote instead of growing linearly forever."""

    controller = DiscountedUCBController(n_arms=1, discount=0.5, warmup_rounds=0)

    for _ in range(50):
        controller.update(mode_id=0, reward=1.0)

    count = controller.get_state()["counts"][0]
    assert count < 3.0


def test_random_controller_samples_only_valid_mode_ids() -> None:
    """Random selection must stay within the valid arm range."""

    controller = RandomController(n_arms=5, random_seed=3)

    samples = [controller.select_mode() for _ in range(100)]

    assert all(0 <= sample < 5 for sample in samples)


def test_seeded_controllers_are_deterministic() -> None:
    """Matching seeds should reproduce the same selection sequences."""

    random_a = RandomController(n_arms=4, random_seed=11)
    random_b = RandomController(n_arms=4, random_seed=11)
    assert [random_a.select_mode() for _ in range(20)] == [
        random_b.select_mode() for _ in range(20)
    ]

    ucb_a = DiscountedUCBController(n_arms=3, warmup_rounds=1, random_seed=19)
    ucb_b = DiscountedUCBController(n_arms=3, warmup_rounds=1, random_seed=19)

    seq_a = []
    seq_b = []
    rewards = [0.0, 1.0, 0.5, 1.0, 0.25]
    for reward in rewards:
        mode_a = ucb_a.select_mode()
        mode_b = ucb_b.select_mode()
        seq_a.append(mode_a)
        seq_b.append(mode_b)
        ucb_a.update(mode_a, reward)
        ucb_b.update(mode_b, reward)

    assert seq_a == seq_b


def test_contextual_controller_initializes_new_bucket_from_global_prior() -> None:
    """Newly visited buckets should inherit the current global discounted-UCB state."""

    controller = BucketedContextualController(
        n_arms=2,
        bucket_names=["stable", "worsening"],
        random_seed=5,
        prior_from_global=True,
    )

    controller.set_context("stable")
    controller.update(mode_id=1, reward=2.0)

    controller.set_context("worsening")
    state = controller.get_state()
    late_bucket_state = state["bucket_states"]["worsening"]

    assert late_bucket_state is not None
    assert late_bucket_state["counts"] == [0.0, 1.0]
    assert late_bucket_state["reward_sums"] == [0.0, 2.0]
    assert state["bucket_visit_counts"]["stable"] == 1
    assert state["bucket_visit_counts"]["worsening"] == 0
