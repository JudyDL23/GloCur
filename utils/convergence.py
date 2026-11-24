# utils/convergence.py
class ConvergenceDetector:
    """Training convergence detector."""

    def __init__(self, threshold=3e-6, patience=3):
        self.threshold = threshold
        self.patience = patience
        self.loss_history = []
        self.no_train_improvement_count = 0
        self.no_val_improvement_count = 0
        self.best_loss = float('inf')

    def check_convergence(self, train_loss, val_loss, best_val_loss):
        """Check whether the training has converged."""
        converged = False

        # Check whether the training loss has improved
        if train_loss < self.best_loss - self.threshold:
            self.best_loss = train_loss
            self.no_train_improvement_count = 0
        else:
            self.no_train_improvement_count += 1

        if self.no_train_improvement_count >= self.patience:
            converged = True
            print(f"No train improvement for {self.patience} epochs")

        # Cond 2:val_loss starts to lift
        if val_loss > best_val_loss:
            self.no_val_improvement_count += 1
        else:
            self.no_val_improvement_count = 0

        if self.no_val_improvement_count >= self.patience:
            converged = True
            print(f"Validation loss increasing for {self.patience} epochs")

        return converged

    def reset(self):
        """Reset the detector state."""
        self.loss_history = []
        self.no_train_improvement_count = 0
        self.no_val_improvement_count = 0
        self.best_loss = float('inf')