# PulseOpt

[![PyPI](https://img.shields.io/pypi/v/pulseopt.svg)](https://pypi.org/project/pulseopt/)
[![Python versions](https://img.shields.io/pypi/pyversions/pulseopt.svg)](https://pypi.org/project/pulseopt/)
[![License](https://img.shields.io/pypi/l/pulseopt.svg)](https://github.com/davidkfoss/pulseopt/blob/main/LICENSE)

**PulseOpt: episodic adaptive control for optimizer dynamics.**

`pulseopt` wraps any PyTorch optimizer with an episode-level bandit that adapts a learning-rate multiplier and a gradient-noise level online. Instead of committing to one static schedule, it evaluates short training episodes ("pulses"), scores them with a shaped log-loss-improvement reward, and picks the next configuration with a discounted-UCB controller. The underlying method is **Adaptive Episodic Exploration Scheduling (AEES)**, exposed as the `AEES` class.

It is small, has a single dependency (`torch>=2.0`), and is designed to drop into an existing training loop with two extra calls per step.

## Install

```bash
pip install pulseopt
```

## Quick start

```python
import torch
from torch import nn
from pulseopt import AEES

model = nn.Linear(8, 4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

aees = AEES(
    optimizer,
    lr_candidates=[0.5, 1.0, 2.0],   # tried as multipliers on the optimizer's base LR
    noise_candidates=[0.0, 0.005],   # tried as gradient-noise std
    episode_length=50,
    lr_scheduler=scheduler,          # optional — AEES calls .step() for you
    seed=0,
)

for step in range(1000):
    aees.step_start(step)            # selects the candidate for this step
    optimizer.zero_grad()
    loss = model(torch.randn(32, 8)).pow(2).mean()
    loss.backward()
    aees.step_end(loss)              # runs optimizer.step() + scheduler.step()

aees.finalize()
logs = aees.get_logs()
print(f"Episodes run: {len(logs['episode_rewards'])}")
print(f"Last selected LR multiplier: {logs['selected_lr_values'][-1]}")
```

The wrapper owns `optimizer.step()` and `lr_scheduler.step()`; you keep `zero_grad()` and `loss.backward()`. The LR multiplier is applied transiently around `optimizer.step()`, so any external scheduler still advances on the optimizer's base learning rate.

## How it works

- **Episode**: a fixed-length window of training steps with one frozen candidate: LR multiplier and/or noise std.
- **Reward**: log-EMA-loss improvement over the episode, minus an optional instability penalty proportional to within-episode loss variance, clipped to `[-1, 1]`.
- **Controller**: discounted-UCB by default; an optional bucketed-contextual variant uses a coarse loss-trend bucket to share information across similar regimes.

Axes with a single candidate are treated as fixed constants and get no controller. Passing `lr_candidates=[1.0]` keeps the LR multiplier disabled, and `noise_candidates=[0.0]` keeps gradient noise off.

## Common knobs

| Argument                    | Meaning                                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------- |
| `lr_candidates`             | Multipliers tried against the optimizer's base LR.                                                  |
| `noise_candidates`          | Gradient-noise std values; `0.0` means no noise.                                                    |
| `episode_length`            | Steps per episode; reward is computed at episode end.                                               |
| `lr_scheduler`              | Optional `torch.optim.lr_scheduler.*` instance; `step()` is called for you.                         |
| `structured_control_mode`   | `"independent"` (default) or `"conditional"` (one noise controller per LR arm).                     |
| `context_mode`              | `"none"` (default) or `"trend"`.                                                                    |
| `reward_instability_lambda` | Weight on the variance penalty in the reward.                                                       |
| `seed`                      | Seeds controllers and gradient-noise generators.                                                    |

`AEES.step_end(loss)` raises `ValueError` on a non-finite loss. If you train with mixed precision (`torch.cuda.amp` / `torch.amp`) and expect occasional NaN/Inf during loss-scaling backoff, guard the call yourself or skip the step.

## Caveats

- AEES does not adapt weight decay; keep it as a normal optimizer hyperparameter.
- Each step clones the optimizer's parameters once to compute an update norm for the reward signal. Memory cost is roughly 1× model size.
- There is no `state_dict` / `load_state_dict` yet — checkpoint and resume are planned for a future minor release.

## Runnable examples

End-to-end demos that use only the public `pulseopt` API (`from pulseopt import AEES`) on real datasets. Each script is short, self-contained, and runs against a `pip install pulseopt`-only environment — no helpers from this repository are imported. Each writes a per-epoch text log to the path given by `--output`.

- [`examples/task_cifar100.py`](https://github.com/davidkfoss/pulseopt/blob/main/examples/task_cifar100.py) — ResNet-18 on CIFAR-100. Picks AdamW or SGD via `--optimizer`. Needs `torch`, `torchvision`.
- [`examples/task_sst2.py`](https://github.com/davidkfoss/pulseopt/blob/main/examples/task_sst2.py) — DistilBERT on GLUE SST-2. AdamW. Needs `torch`, `transformers`, `datasets`.
- [`examples/task_agnews.py`](https://github.com/davidkfoss/pulseopt/blob/main/examples/task_agnews.py) — DistilBERT on AG News. AdamW. Needs `torch`, `transformers`, `datasets`.

```bash
pip install pulseopt torch torchvision
python examples/task_cifar100.py --epochs 10 --output cifar100.log
```

These are the recommended starting point if you want to see how AEES plugs into a normal training loop.

## Repo layout

- [`src/pulseopt/`](https://github.com/davidkfoss/pulseopt/tree/main/src/pulseopt) — published library: controllers, episode manager, reward, optimizer wrappers, and the `AEES` high-level API.
- [`examples/`](https://github.com/davidkfoss/pulseopt/tree/main/examples) — short, self-contained PyPI-side demos using the public `AEES` API.
- [`tests/`](https://github.com/davidkfoss/pulseopt/tree/main/tests) — regression and unit tests.

## Development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .[dev,examples]
pytest
```

## License

MIT — see [LICENSE](https://github.com/davidkfoss/pulseopt/blob/main/LICENSE).
