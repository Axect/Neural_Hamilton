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

# ┌──────────────────────────────────────────────────────────┐
#  Improved Predicted Final Loss (PFL) Pruner V2
# └──────────────────────────────────────────────────────────┘
class PFLPrunerV2(BasePruner):
    """
    Improved Predicted Final Loss (PFL) based pruner.

    This pruner models learning curves using a power-law fit (y = a*x^b)
    and prunes a trial if its predicted final loss is worse than the
    actual final loss of the k-th best completed trial.
    """

    def __init__(
        self,
        n_startup_trials: int = 10,
        n_warmup_epochs: int = 10,
        top_k: int = 10,
        target_epoch: int = 50,
        min_points_for_prediction: int = 3,
    ):
        super().__init__()
        self.n_startup_trials = n_startup_trials
        self.n_warmup_epochs = n_warmup_epochs
        self.top_k = top_k
        self.target_epoch = target_epoch
        self.min_points_for_prediction = min_points_for_prediction

        self.top_k_final_losses: List[float] = []
        self.completed_trials_count = 0

    def complete_trial(self, trial_id: int) -> None:
        if trial_id in self._trials:
            trial = self._trials[trial_id]
            final_loss = self._get_final_loss(trial)

            if np.isfinite(final_loss):
                self.completed_trials_count += 1
                if len(self.top_k_final_losses) < self.top_k:
                    bisect.insort(self.top_k_final_losses, final_loss)
                elif final_loss < self.top_k_final_losses[-1]:
                    self.top_k_final_losses.pop()
                    bisect.insort(self.top_k_final_losses, final_loss)

            super().complete_trial(trial_id)
            del self._trials[trial_id]

    def _get_final_loss(self, trial: Trial) -> float:
        """Get the average final loss across all seeds for a completed trial."""
        if not trial.seed_values:
            return float("inf")
        
        total_loss = 0.0
        n_seeds = len(trial.seed_values)
        for loss_vec in trial.seed_values.values():
            if not loss_vec: return float("inf")
            total_loss += loss_vec[-1]
        
        return total_loss / n_seeds if n_seeds > 0 else float("inf")

    def _predict_final_loss_power_law(self, losses: List[float]) -> float:
        """
        Predict final loss using power-law curve fitting (y = a*x^b).
        This is equivalent to a linear fit in log-log space.
        """
        n_losses = len(losses)
        if n_losses < self.min_points_for_prediction:
            return float("inf")

        try:
            # x: epochs (1-based), y: losses
            epochs = np.arange(1, n_losses + 1)
            # Clip losses to avoid log(0) issues
            safe_losses = np.maximum(losses, 1e-10)

            log_epochs = np.log(epochs)
            log_losses = np.log(safe_losses)

            # Linear fit in log-log space
            b, log_a = np.polyfit(log_epochs, log_losses, 1)

            # Prune if the slope (b) is positive
            if b > 0:
                return float("inf")

            # Predict final loss at target_epoch
            predicted_log_loss = log_a + b * np.log(self.target_epoch)
            predicted_loss = np.exp(predicted_log_loss)
            
            # Return the minimum of predicted loss and the actual final losses
            return min(predicted_loss, min(losses))

        except (np.linalg.LinAlgError, ValueError):
            # If fitting fails, return a large value to indicate pruning
            return float("inf")

    def _should_prune_trial(self, trial: Trial) -> bool:
        # Check if any seed has invalid loss
        for losses in trial.seed_values.values():
            if not losses or not np.isfinite(losses[-1]):
                return True

        # Don't prune during warmup period
        if (
            self.completed_trials_count < self.n_startup_trials
            or trial.current_epoch <= self.n_warmup_epochs
        ):
            return False

        # Don't prune if we have not enough top_k final losses
        if len(self.top_k_final_losses) < 1:
            return False

        avg_predicted_loss = 0.0
        n_seeds = len(trial.seed_values)
        if n_seeds == 0: return False

        for loss_vec in trial.seed_values.values():
            avg_predicted_loss += self._predict_final_loss_power_law(loss_vec)
        
        avg_predicted_loss /= n_seeds

        # Get the worst final loss from the top k trials
        pruning_threshold = self.top_k_final_losses[-1]

        # Prune if the predicted final loss is worse than the threshold
        return avg_predicted_loss > pruning_threshold
