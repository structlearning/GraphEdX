import os
import torch

class EarlyStoppingModule(object):
    """
    Module to keep track of validation score across epochs
    Stop training if score not imroving exceeds patience
    """

    def __init__(
        self, save_dir=".", task_name="TASK", patience=100, delta=0.0001, logger=None
    ):
        self.save_dir = save_dir
        self.task_name = task_name
        self.patience = patience
        self.delta = delta
        self.logger = logger
        self.create_dirs()
        self.best_scores = None
        self.num_bad_epochs = 0
        self.should_stop_now = False

    def create_dirs(self):
        # Initial
        save_dir = os.path.join(self.save_dir, "initialModels")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.initial_model_path = os.path.join(save_dir, self.task_name)

        # Latest
        save_dir = os.path.join(self.save_dir, "latestModels")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.latest_model_path = os.path.join(save_dir, self.task_name)

        # Best
        save_dir = os.path.join(self.save_dir, "bestValidationModels")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.best_model_path = os.path.join(save_dir, self.task_name)

    def save_initial_model(self, model):
        self.logger.info(f"saving initial model to {self.initial_model_path}")
        output = open(self.initial_model_path, mode="wb")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            output,
        )
        output.close()

    def save_latest_model(self, model, epoch, optimizer):
        output = open(self.latest_model_path, mode="wb")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "patience": self.patience,
                "best_scores": self.best_scores,
                "num_bad_epochs": self.num_bad_epochs,
                "should_stop_now": self.should_stop_now,
                "optim_state_dict": optimizer.state_dict(),
            },
            output,
        )
        output.close()

    def load_latest_model(self):
        if not os.path.exists(self.latest_model_path):
            return None

        self.logger.info(f"loading latest trained model from {self.latest_model_path}",)
        checkpoint = torch.load(self.latest_model_path)
        self.patience = checkpoint["patience"]
        self.best_scores = checkpoint["best_scores"]
        self.num_bad_epochs = checkpoint["num_bad_epochs"]
        self.should_stop_now = checkpoint["should_stop_now"]
        return checkpoint

    def save_best_model(self, model, epoch):
        self.logger.info(f"saving best validated model to {self.best_model_path}")
        output = open(self.best_model_path, mode="wb")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
            },
            output,
        )
        output.close()

    def load_best_model(self):
        self.logger.info(f"loading best validated model from {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path)
        return checkpoint

    def diff(self, curr_scores):
        return sum([cs - bs for cs, bs in zip(curr_scores, self.best_scores)])

    def check(self, curr_scores, model, epoch, optimizer):
        if self.best_scores is None:
            self.best_scores = curr_scores
            self.save_best_model(model, epoch)
        elif self.diff(curr_scores) >= self.delta:
            self.num_bad_epochs = 0
            self.best_scores = curr_scores
            self.save_best_model(model, epoch)
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self.should_stop_now = True
        self.save_latest_model(model, epoch, optimizer)
        return self.should_stop_now
