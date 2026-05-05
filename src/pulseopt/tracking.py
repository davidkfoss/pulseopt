"""Lightweight tracking primitives shared by episode management."""

from __future__ import annotations

import math


class EMATracker:
    """Track an exponential moving average for scalar values."""

    def __init__(
        self,
        alpha: float,
        initial_value: float | None = None,
        eps: float = 1e-12,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1].")
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        if initial_value is not None and not math.isfinite(initial_value):
            raise ValueError("initial_value must be finite when provided.")

        self.alpha = alpha
        self.eps = eps
        self._value = initial_value

    @property
    def value(self) -> float | None:
        """Return the current EMA value, if initialized."""

        return self._value

    @property
    def initialized(self) -> bool:
        """Whether at least one value has been observed."""

        return self._value is not None

    def update(self, value: float) -> float:
        """Update the EMA with a new scalar observation."""

        if not math.isfinite(value):
            raise ValueError("EMA update value must be finite.")

        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        return self._value
