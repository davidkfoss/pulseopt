"""Small approximate FLOP helpers for thesis-scale experiment comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


_RESNET18_CIFAR100_FORWARD_FLOPS_PER_EXAMPLE = 3.7e7
_DISTILBERT_LAYERS = 6
_DISTILBERT_HIDDEN_SIZE = 768
_DISTILBERT_FF_SIZE = 3072
_DISTILBERT_NUM_LABELS = 2
_TRAIN_MULTIPLIER = 3.0


def estimate_resnet18_cifar100_train_flops_per_batch(batch_size: int) -> float:
    """Approximate CIFAR ResNet-18 training FLOPs as 3x forward cost."""

    _validate_batch_size(batch_size)
    return batch_size * _RESNET18_CIFAR100_FORWARD_FLOPS_PER_EXAMPLE * _TRAIN_MULTIPLIER


def estimate_resnet18_cifar100_eval_flops_per_batch(batch_size: int) -> float:
    """Approximate CIFAR ResNet-18 evaluation FLOPs using forward cost only."""

    _validate_batch_size(batch_size)
    return batch_size * _RESNET18_CIFAR100_FORWARD_FLOPS_PER_EXAMPLE


def estimate_distilbert_sst2_train_flops_per_batch(batch_size: int, seq_len: int) -> float:
    """Approximate DistilBERT SST-2 training FLOPs as 3x forward cost."""

    return estimate_distilbert_sst2_eval_flops_per_batch(
        batch_size=batch_size,
        seq_len=seq_len,
    ) * _TRAIN_MULTIPLIER


def estimate_distilbert_sst2_eval_flops_per_batch(batch_size: int, seq_len: int) -> float:
    """Approximate DistilBERT SST-2 forward FLOPs with a readable Transformer formula."""

    _validate_batch_size(batch_size)
    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer.")

    hidden = float(_DISTILBERT_HIDDEN_SIZE)
    feedforward = float(_DISTILBERT_FF_SIZE)
    tokens = float(seq_len)

    attention_projections = 4.0 * tokens * hidden * hidden
    attention_scores_and_mix = 2.0 * tokens * tokens * hidden
    feedforward_blocks = 2.0 * tokens * hidden * feedforward
    per_layer_forward = (
        attention_projections + attention_scores_and_mix + feedforward_blocks
    )

    embedding_and_classifier = (
        2.0 * tokens * hidden + 2.0 * hidden * float(_DISTILBERT_NUM_LABELS)
    )
    per_example_forward = (
        float(_DISTILBERT_LAYERS) * per_layer_forward + embedding_and_classifier
    )
    return batch_size * per_example_forward


@dataclass
class FlopAccumulator:
    """Accumulate approximate train/eval FLOPs and epoch-level TFLOP summaries."""

    train_flops: float = 0.0
    eval_flops: float = 0.0
    _current_epoch_flops: float = 0.0
    _epoch_tflops: list[float] = field(default_factory=list)

    def add_train_batch(self, flops: float) -> None:
        """Add one batch of training FLOPs."""

        validated = _validate_flops(flops)
        self.train_flops += validated
        self._current_epoch_flops += validated

    def add_eval_batch(self, flops: float) -> None:
        """Add one batch of evaluation FLOPs."""

        validated = _validate_flops(flops)
        self.eval_flops += validated
        self._current_epoch_flops += validated

    def finish_epoch(self) -> float:
        """Close the current epoch and return its TFLOP total."""

        epoch_tflops = self._current_epoch_flops / 1e12
        self._epoch_tflops.append(epoch_tflops)
        self._current_epoch_flops = 0.0
        return epoch_tflops

    @property
    def total_flops(self) -> float:
        """Return total accumulated FLOPs across train and eval."""

        return self.train_flops + self.eval_flops

    @property
    def total_tflops(self) -> float:
        """Return total accumulated FLOPs expressed in TFLOPs."""

        return self.total_flops / 1e12

    @property
    def epoch_tflops(self) -> list[float]:
        """Return the per-epoch TFLOP totals collected so far."""

        return list(self._epoch_tflops)


def _validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")


def _validate_flops(flops: float) -> float:
    if not math.isfinite(flops) or flops < 0.0:
        raise ValueError("flops must be a finite non-negative float.")
    return float(flops)
