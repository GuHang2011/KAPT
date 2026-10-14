"""Validated early stopping based exclusively on a validation metric."""

from math import isfinite
from numbers import Integral, Real
from typing import Literal


def _finite_number(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(message) from exc
    if not isfinite(converted):
        raise ValueError(message)
    return converted


class EarlyStopping:
    """Stop after ``patience`` consecutive observations without improvement.

    Improvement is strict: a change equal to ``min_delta`` is insufficient.
    Epochs must be nonnegative and strictly increase. Once stopped, the decision
    remains true until ``reset``; later observations cannot change the best score.
    """

    def __init__(
        self, patience: int = 10, min_delta: float = 0.0,
        mode: Literal["min", "max"] = "max",
    ) -> None:
        if isinstance(patience, bool) or not isinstance(patience, Integral) or patience < 1:
            raise ValueError("patience must be a positive integer")
        min_delta = _finite_number(min_delta, "min_delta must be a finite nonnegative number")
        if min_delta < 0:
            raise ValueError("min_delta must be a finite nonnegative number")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.last_epoch: int | None = None
        self.bad_epochs = 0
        self.stopped = False

    def update(self, score: float, epoch: int) -> bool:
        """Observe a score and return whether training should stop."""
        score = _finite_number(score, "score must be a finite number")
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if self.last_epoch is not None and epoch <= self.last_epoch:
            raise ValueError("epochs must strictly increase")
        self.last_epoch = int(epoch)
        if self.stopped:
            return True
        improvement = self.best_score is None
        if self.best_score is not None:
            improvement = (
                score > self.best_score + self.min_delta if self.mode == "max"
                else score < self.best_score - self.min_delta
            )
        if improvement:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.stopped = self.bad_epochs >= self.patience
        return self.stopped

    def __call__(self, score: float, epoch: int) -> bool:
        return self.update(score, epoch)
