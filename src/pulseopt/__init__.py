"""Adaptive Episodic Exploration Scheduling package."""

__version__ = "0.1.3"

from pulseopt.scheduler import AEES
from pulseopt.optimizer import AdaptiveModeAdamW, AdaptiveModeSGD
from pulseopt.types import CandidateConfig, EpisodeSummary, StructuredSelection
from pulseopt.episode import StructuredEpisodeManager
from pulseopt.controller import (
    BaseController,
    BucketedContextualController,
    DiscountedUCBController,
    RandomController,
    TREND_CONTEXT_BUCKETS,
    TREND_PHASE_CONTEXT_BUCKETS,
)
from pulseopt.reward import BaseReward, NormalizedLossImprovementReward
from pulseopt.modes import build_candidate_configs_from_axes

__all__ = [
    # High-level API
    "AEES",
    # Optimizer wrappers
    "AdaptiveModeAdamW",
    "AdaptiveModeSGD",
    # Types
    "CandidateConfig",
    "EpisodeSummary",
    "StructuredSelection",
    # Episode managers
    "StructuredEpisodeManager",
    # Controllers
    "BaseController",
    "BucketedContextualController",
    "DiscountedUCBController",
    "RandomController",
    "TREND_CONTEXT_BUCKETS",
    "TREND_PHASE_CONTEXT_BUCKETS",
    # Reward
    "BaseReward",
    "NormalizedLossImprovementReward",
    # Candidate helpers
    "build_candidate_configs_from_axes",
]
