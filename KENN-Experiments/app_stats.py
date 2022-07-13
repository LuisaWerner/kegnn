
class RunStats(object):
    def __init__(self, run: int,
                   train_losses: list,
                   train_accuracies: list,
                   valid_losses: list,
                   valid_accuracies: list,
                   test_acc: float,
                   epoch_time: list):
        self.run = run
        self.train_losses = train_losses
        self.train_accuracies = train_accuracies
        self.valid_losses = valid_losses
        self.valid_accuracies = valid_accuracies
        self.test_accuracy = test_acc
        self.epoch_time = epoch_time

        self.max_valid_accuracy = max(valid_accuracies)
        self.max_train_accuracy = max(train_accuracies)
        self.avg_epoch_time = sum(epoch_time)/len(epoch_time)

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

    def add_run(self, run_stats:RunStats):
        self.run_stats.append(run_stats)

    def end_experiment(self):
        """computes experiment statistics"""
        max_train_accuracies = [rs.max_train_accuracy for rs in self.run_stats]
        max_valid_accuracies = [rs.max_valid_accuracy for rs in self.run_stats]


print(RunStats(0,[9],[9],[9],[9],0,[9]))