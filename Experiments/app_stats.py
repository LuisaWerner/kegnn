import statistics

import numpy as np


class RunStats(object):
    def __init__(self, run: int,
                 train_losses: list,
                 train_accuracies: list,
                 valid_losses: list,
                 valid_accuracies: list,
                 test_acc: float,
                 epoch_time: list,
                 test_accuracies: list):
        self.run = run
        self.train_losses = train_losses
        self.train_accuracies = train_accuracies
        self.valid_losses = valid_losses
        self.valid_accuracies = valid_accuracies
        self.test_accuracy = test_acc
        self.epoch_time = epoch_time
        self.test_accuracies = test_accuracies

        self.max_valid_accuracy = max(valid_accuracies, default=0)
        self.max_train_accuracy = max(train_accuracies, default=0)
        try:
            self.avg_epoch_time = sum(epoch_time) / len(epoch_time)
        except ZeroDivisionError:
            self.avg_epoch_time = None

    def to_dict(self):
        return {
            "run": self.run,
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "valid_losses": self.valid_losses,
            "valid_accuracies": self.valid_accuracies,
            "test_accuracy": self.test_accuracy,
            "epoch_time": self.epoch_time,
            "max_valid_accuracy": self.max_valid_accuracy,
            "max_train_accuracy": self.max_train_accuracy,
            "avg_run_epoch_time": self.avg_epoch_time
        }

    def __str__(self):
        return (f"Results of run {self.run}:\n"
                f"Maximum accuracy on train: {self.max_train_accuracy}\n"
                f"Maximum accuracy on valid: {self.max_valid_accuracy}\n"
                f"Accuracy on test: {self.test_accuracy}\n"
                f"Avg epoch time: {self.avg_epoch_time}\n")


class ExperimentStats(object):
    def __init__(self):
        self.run_stats = []
        self.avg_test_accuracy = -1
        self.avg_train_accuracy = -1
        self.avg_valid_accuracy = -1
        self.highest_train_accuracy = -1
        self.highest_valid_accuracy = -1
        self.avg_epoch_time = -1
        self.sd_test_accuracy = -1

    def add_run(self, run_stats: RunStats):
        self.run_stats.append(run_stats)

    def end_experiment(self):
        """computes experiment statistics"""
        max_train_accuracies = [rs.max_train_accuracy for rs in self.run_stats]
        max_valid_accuracies = [rs.max_valid_accuracy for rs in self.run_stats]
        self.avg_train_accuracy = np.mean(max_train_accuracies)
        self.highest_train_accuracy = max(max_train_accuracies)
        self.avg_valid_accuracy = np.mean(max_valid_accuracies)
        self.highest_valid_accuracy = max(max_valid_accuracies)
        self.avg_epoch_time = np.mean([rs.avg_epoch_time for rs in self.run_stats])
        accuracies = [rs.test_accuracy for rs in self.run_stats]
        self.avg_test_accuracy = np.mean(accuracies)
        if len(accuracies) > 1:
            self.sd_test_accuracy = statistics.stdev(accuracies)

    def to_dict(self):
        return {
            "avg_train_accuracy": self.avg_train_accuracy,
            "avg_valid_accuracy": self.avg_valid_accuracy,
            "avg_test_accuracy": self.avg_test_accuracy,
            "highest_valid_accuracy": self.highest_valid_accuracy,
            "highest_train_accuracy": self.highest_train_accuracy,
            "avg_epoch_time": self.avg_epoch_time,
            "sd_test_accuracy": self.sd_test_accuracy
        }

    def __str__(self):
        runs = len(self.run_stats)
        return (f"Average accuracy over {runs} iterations  on train :{self.avg_train_accuracy}\n"
                f"Average accuracy over {runs} iterations on valid :{self.avg_valid_accuracy}\n"
                f"Average test accuracy over {runs} iterations :{self.avg_test_accuracy}\n"
                f"Highest accuracy over train: {self.highest_train_accuracy}\n"
                f"Highest accuracy over valid: {self.highest_valid_accuracy}\n"
                f"Average epoch time: {self.avg_epoch_time}")
