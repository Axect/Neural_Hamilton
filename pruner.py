import numpy as np
from typing import List, Dict, Tuple, Optional
import bisect
from dataclasses import dataclass, field
from util import predict_final_loss


# ┌──────────────────────────────────────────────────────────┐
#  Trial Dataclass
# └──────────────────────────────────────────────────────────┘
@dataclass
class Trial:
    """Trial class to hold intermediate state."""

    trial_id: int
    current_epoch: int = 0
    seed_values: Dict[int, List[float]] = field(default_factory=dict)

    def add_value(self, seed: int, value: float) -> None:
        """Add a new intermediate value for a given seed."""
        if seed not in self.seed_values:
            self.seed_values[seed] = []
        self.seed_values[seed].append(value)
        self.current_epoch = len(self.seed_values[seed])


# ┌──────────────────────────────────────────────────────────┐
#  Base Pruner Class
# └──────────────────────────────────────────────────────────┘
class BasePruner:
    """
    Pruner base class with Optuna-like interface.
    """

    def __init__(self):
        self._trials: Dict[int, Trial] = {}
        self._current_trial: Optional[Trial] = None

    def register_trial(self, trial_id: int) -> None:
        """Register a new trial."""
        self._trials[trial_id] = Trial(trial_id=trial_id)

    def complete_trial(self, trial_id: int) -> None:
        """Mark a trial as finished and clean up."""
        if trial_id in self._trials:
            if self._current_trial and self._current_trial.trial_id == trial_id:
                self._current_trial = None
            del self._trials[trial_id]

    def report(self, trial_id: int, seed: int, epoch: int, value: float) -> None:
        """Report an intermediate value for a given trial.

        Args:
            trial_id: Trial identifier
            seed: Random seed being used
            epoch: Current epoch number
            value: Intermediate value to report (typically validation loss)
        """
        if trial_id not in self._trials:
            self.register_trial(trial_id)

        trial = self._trials[trial_id]
        trial.add_value(seed, value)
        self._current_trial = trial

    def should_prune(self) -> bool:
        """Decide whether the current trial should be pruned at the current step.

        Returns:
            bool: True if the trial should be pruned
        """
        if not self._current_trial:
            return False
        return self._should_prune_trial(self._current_trial)

    def _should_prune_trial(self, trial: Trial) -> bool:
        """Implementation specific pruning logic."""
        raise NotImplementedError


# ┌──────────────────────────────────────────────────────────┐
#  Predicted Final Loss (PFL) Pruner
# └──────────────────────────────────────────────────────────┘
class PFLPruner(BasePruner):
    """Predicted Final Loss (PFL) based pruner with Optuna-like interface.

    This pruner maintains top k trials based on validation loss and prunes trials
    if their predicted final loss is worse than the worst saved PFL.
    """

    def __init__(
        self,
        n_startup_trials: int = 10,
        n_warmup_epochs: int = 10,
        top_k: int = 10,
        target_epoch: int = 50,
    ):
        super().__init__()
        self.n_startup_trials = n_startup_trials
        self.n_warmup_epochs = n_warmup_epochs
        self.top_k = top_k
        self.target_epoch = target_epoch

        self.top_pairs: List[Tuple[float, float]] = []  # List of (val_loss, pfl) pairs
        self.completed_trials = 0

    def complete_trial(self, trial_id: int) -> None:
        """Mark a trial as finished and check for inclusion in top-k."""
        if trial_id in self._trials:
            self.completed_trials += 1
            self._check_and_insert(self._trials[trial_id])
            super().complete_trial(trial_id)

    def _check_and_insert(self, trial: Trial) -> None:
        """Check if a trial should be inserted into top k and insert if needed."""
        val_loss, pfl = self._compute_trial_metrics(trial)
        if self._should_insert_pair(val_loss):
            self._insert_pair(val_loss, pfl)

    def _compute_trial_metrics(self, trial: Trial) -> Tuple[float, float]:
        """Compute average val_loss and PFL for a trial across all seeds."""
        if not trial.seed_values:
            return float("inf"), -float("inf")

        # Average the last val_loss and PFL across seeds
        avg_val_loss = 0.0
        avg_pfl = 0.0
        n_seeds = len(trial.seed_values)

        for loss_vec in trial.seed_values.values():
            if loss_vec:  # Check if there are any losses for this seed
                avg_val_loss += loss_vec[-1]  # Last validation loss
                avg_pfl += self._predict_final_loss(loss_vec)

        avg_val_loss /= n_seeds
        avg_pfl /= n_seeds
        return avg_val_loss, avg_pfl

    def _predict_final_loss(self, losses: List[float]) -> float:
        """Predict final loss value using the loss history."""
        if len(losses) < 2:
            return -float("inf")

        try:
            return (
                -np.log10(losses[-1])
                if len(losses) < 10
                else predict_final_loss(losses, self.target_epoch)
            )
        except:
            return -float("inf")

    def _should_insert_pair(self, val_loss: float) -> bool:
        """Check if a new pair should be inserted based on validation loss."""
        if len(self.top_pairs) < self.top_k:
            return True
        return val_loss < self.top_pairs[-1][0]

    def _insert_pair(self, val_loss: float, pfl: float) -> None:
        """Insert a new (val_loss, pfl) pair maintaining sorted order."""
        pair = (val_loss, pfl)

        # Find insertion point using binary search
        idx = bisect.bisect_left(self.top_pairs, pair)

        # Insert the pair
        if len(self.top_pairs) < self.top_k:
            self.top_pairs.insert(idx, pair)
        elif idx < self.top_k:
            self.top_pairs.insert(idx, pair)
            self.top_pairs.pop()  # Remove worst pair if we exceed top_k

    def _should_prune_trial(self, trial: Trial) -> bool:
        """Implementation of trial pruning logic."""
        # Check if any seed has invalid loss
        for losses in trial.seed_values.values():
            if not losses or not np.isfinite(losses[-1]):
                return True

        # Don't prune during warmup period
        if (
            self.completed_trials < self.n_startup_trials
            or trial.current_epoch <= self.n_warmup_epochs
        ):
            return False

        # Compute current metrics
        _, curr_pfl = self._compute_trial_metrics(trial)

        # Prune if PFL is worse than all saved PFLs
        if self.top_pairs:  # Only if we have recorded pairs
            worst_pfl = min(pair[1] for pair in self.top_pairs)
            return curr_pfl < worst_pfl

        return False
