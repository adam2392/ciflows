import os

import torch


class TopKModelSaver:
    def __init__(self, save_dir, k=5):
        self.save_dir = save_dir
        self.k = k
        self.best_models = []  # List of tuples (loss, model_state_dict, epoch)

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

    def check(self, loss):
        """Determine if the current model's loss is worth saving."""
        # Check if the current model's loss should be saved
        if len(self.best_models) < self.k:
            return True  # If we have fewer than k models, always save the model
        else:
            # If the current loss is better than the worst model's loss, return True
            if loss < self.best_models[-1][0]:
                return True
            else:
                return False

    def save_model(self, model, epoch, loss):
        """Save the model if it's among the top-k based on the training loss."""
        # First, check if the model should be saved
        if self.check(loss):
            # If we have fewer than k models, simply append the model
            if len(self.best_models) < self.k:
                self.best_models.append((loss, epoch))
            else:
                # If the current loss is better than the worst model, replace it
                self.best_models.append((loss, epoch))

            # Sort by loss (ascending order) and remove worse models if necessary
            self.best_models.sort(key=lambda x: x[0])  # Sort by loss (ascending)

            # Save the model
            self._save_model(model, epoch, loss)

            # Remove worse models if there are more than k models
            self.remove_worse_models()

    def _save_model(self, model, epoch, loss):
        """Helper function to save the model to disk."""
        filename = os.path.join(self.save_dir, f"model_epoch_{epoch}.pt")
        # Save the model state_dict
        torch.save(model.state_dict(), filename)
        print(f"Saved model to {filename}")

    def remove_worse_models(self):
        """Remove the worse models if there are more than k models."""
        # Ensure the list is sorted by the loss (ascending order)
        self.best_models.sort(key=lambda x: x[0])  # Sort by loss (ascending)

        # Remove models beyond the top-k
        while len(self.best_models) > self.k:
            loss, epoch = self.best_models.pop()
            filename = os.path.join(self.save_dir, f"model_epoch_{epoch}.pt")
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Removed worse model {filename}")
