"""Small typed helpers for AEES experiment run plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any


_ARG_ORDER = [
    "method",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "lr_scheduler",
    "scheduler_t_max",
    "warmup_epochs",
    "episode_length",
    "lr_candidates",
    "noise_candidates",
    "structured_control_mode",
    "context_mode",
    "context_trend_window",
    "context_trend_epsilon",
    "reward_epsilon",
    "reward_instability_lambda",
    "reward_clip_min",
    "reward_clip_max",
    "label_noise_type",
    "label_noise_rate",
    "max_length",
    "seed",
    "output",
]


@dataclass(frozen=True)
class RunSpec:
    """One generated experiment command."""

    name: str
    script: str
    args: dict[str, object]
    output_path: str
    tags: list[str]


def build_cli_args(args: dict[str, object]) -> list[str]:
    """Convert one argument dictionary into deterministic CLI tokens."""

    ordered_keys = _ordered_arg_keys(args)
    cli_args: list[str] = []
    for key in ordered_keys:
        value = args[key]
        if value is None or value is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            cli_args.append(flag)
            continue
        cli_args.extend([flag, _stringify_arg_value(value)])
    return cli_args


def runspec_to_command(
    spec: RunSpec,
    python_executable: str = "python3.11",
) -> list[str]:
    """Convert one run spec into a full command list."""

    return [python_executable, spec.script, *build_cli_args(spec.args)]


def runspec_to_dict(spec: RunSpec) -> dict[str, object]:
    """Convert one run spec into a manifest-friendly dictionary."""

    payload = asdict(spec)
    payload["command"] = runspec_to_command(spec)
    payload["command_str"] = " ".join(_shell_quote(token) for token in payload["command"])
    return payload


def write_manifest(specs: list[RunSpec], output_path: str | Path) -> None:
    """Persist one run manifest as stable JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_count": len(specs),
        "runs": [runspec_to_dict(spec) for spec in specs],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def slugify(value: str) -> str:
    """Return a filesystem-safe lowercase token."""

    lowered = value.strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "_", lowered)
    return compact.strip("_")


def join_slug_parts(*parts: object) -> str:
    """Join slug tokens while skipping empty parts."""

    tokens = [slugify(str(part)) for part in parts if str(part).strip()]
    return "_".join(token for token in tokens if token)


def _ordered_arg_keys(args: dict[str, object]) -> list[str]:
    preferred_keys = [key for key in _ARG_ORDER if key in args]
    remaining_keys = sorted(key for key in args if key not in _ARG_ORDER)
    return [*preferred_keys, *remaining_keys]


def _stringify_arg_value(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return ",".join(_format_scalar(item) for item in value)
    return _format_scalar(value)


def _format_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _shell_quote(token: str) -> str:
    if token and re.fullmatch(r"[A-Za-z0-9_./:=,-]+", token):
        return token
    escaped = token.replace("'", "'\"'\"'")
    return f"'{escaped}'"
