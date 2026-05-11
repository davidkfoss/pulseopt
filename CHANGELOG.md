# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-11

### Deprecated
- `AdaptiveModeAdamW` and `AdaptiveModeSGD` are deprecated and will be removed
  in 0.4.0. Use `AEES`, which wraps any `torch.optim.Optimizer`. Importing or
  instantiating either class now emits a `DeprecationWarning`.

## [0.2.2] - 2026-05-10

Patch release with packaging cleanup and security hardening.

### Added
- `MANIFEST.in` to refine PyPI source distribution contents.
- Runnable examples included in the source distribution (wheel kept minimal).

### Changed
- Clarified README documentation around examples and repository layout.
- Updated test configuration.
- Pinned third-party GitHub Actions to commit SHAs to reduce CI supply-chain risk.

### Fixed
- Replaced an unsafe `assert` state check in `Scheduler.step_end` with an
  explicit runtime check.

## [0.2.1] - 2026-05-07

Metadata-only patch.

### Added
- `Changelog` URL under `[project.urls]` so PyPI surfaces a link to the
  changelog in the project sidebar.

## [0.2.0] - 2026-05-07

Breaking-change release that removes the `trend_phase` context mode.

### Removed
- `pulseopt.TREND_PHASE_CONTEXT_BUCKETS`.
- `context_mode="trend_phase"`.
- `total_training_steps` kwarg on `AEES` and `StructuredEpisodeManager` (was
  only used to compute the `trend_phase` boundary).
- `StructuredSelection.context_bucket_id`, `.context_bucket_name`,
  `.context_phase`, `.context_trend`.
- Episode log keys `context_bucket_ids`, `context_bucket_names`,
  `context_phases`, `context_trends`.

### Changed
- `StructuredSelection` now exposes a single `context_bucket: str | None`
  field; episode logs expose a single `context_buckets` list (only present
  when `context_mode != "none"`).

### Migration
- If you used `context_mode="trend_phase"`, switch to `"trend"` (or `"none"`).
- If you read any of the removed `StructuredSelection` fields or log keys,
  read `context_bucket` / `context_buckets` instead.

## [0.1.6rc1] - 2026-05-06

### Added
- CI/CD workflows and ruff lint/format configuration.
- Release smoke test for the CI/CD pipeline.

## [0.1.5] - 2026-05-06

### Removed
- Files irrelevant to the PyPI package that remained from thesis experiments.

### Changed
- Default `reward_instability_lambda` set to `0.0` in `reward.py`.

## [0.1.4] - 2026-05-05

### Fixed
- Changed repo-relative paths to absolute.

### Changed
- Set reward instability constant to `0.0`.

[Unreleased]: https://github.com/davidkfoss/pulseopt/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/davidkfoss/pulseopt/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/davidkfoss/pulseopt/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/davidkfoss/pulseopt/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/davidkfoss/pulseopt/compare/v0.1.6rc1...v0.2.0
[0.1.6rc1]: https://github.com/davidkfoss/pulseopt/compare/v0.1.5...v0.1.6rc1
[0.1.5]: https://github.com/davidkfoss/pulseopt/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/davidkfoss/pulseopt/releases/tag/v0.1.4
