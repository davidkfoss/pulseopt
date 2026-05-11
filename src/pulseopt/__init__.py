"""Adaptive Episodic Exploration Scheduling package."""

__version__ = "0.3.0"

from pulseopt.controller import (
    TREND_CONTEXT_BUCKETS,
    BaseController,
    BucketedContextualController,
    DiscountedUCBController,
    RandomController,
)
from pulseopt.episode import StructuredEpisodeManager
from pulseopt.modes import build_candidate_configs_from_axes
from pulseopt.optimizer import AdaptiveModeAdamW, AdaptiveModeSGD
from pulseopt.reward import BaseReward, NormalizedLossImprovementReward
from pulseopt.scheduler import AEES
from pulseopt.types import CandidateConfig, EpisodeSummary, StructuredSelection

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
    # Reward
    "BaseReward",
    "NormalizedLossImprovementReward",
    # Candidate helpers
    "build_candidate_configs_from_axes",
]
