import os
import pickle

import numpy as np
import torch


class Logger(object):
    def __init__(self, name, args):
        # todo: change for several models
        self.name = name
        self.results = {}
        self.results.setdefault(name, [])
        self.es_min_delta = args.es_min_delta
        self.es_patience = args.es_patience

    def add_result(self, train_losses: list,
                   train_accuracies: list,
                   valid_losses: list,
                   valid_accuracies: list,
                   test_acc: float,
                   run: int,
                   clause_weights_dict=None):
        """
        adds the losses and accuracies of a run to the results dictionary
        """
        run_results = {'train_losses': train_losses,
                       'train_accuracies': train_accuracies,
                       'valid_losses': valid_losses,
                       'valid_accuracies': valid_accuracies,
                       'clause_weights_dict': clause_weights_dict,
                       'test_accuracy': test_acc
                       }
        self.results[self.name].append(run_results)
        self.print_results_run(run)

    def print_results_run(self, run: int):
        """ Prints results after all epochs per run """
        max_valid_acc = max(self.results[self.name][run]['valid_accuracies'])
        max_train_acc = max(self.results[self.name][run]['train_accuracies'])
        print(f"Results of run {run}:")
        print(f"Maximum accuracy on train: {max_train_acc}")
        print(f"Maximum accuracy on valid: {max_valid_acc}")
        print(f"Accuracy on test: {self.results[self.name][run]['test_accuracy']}")

    def print_results(self, args, setting: str):
        """ Prints results after all runs """
        max_epoch_acc_train = []
        max_epoch_acc_valid = []
        for run in range(len(self.results[self.name])):
            max_epoch_acc_train.append(max(self.results[self.name][run]['train_accuracies']))
            max_epoch_acc_valid.append(max(self.results[self.name][run]['valid_accuracies']))

        print(f"Results of {setting} training, {args.runs} runs, {args.epochs} epochs ")
        print(f"Average accuracy over {args.runs} iterations  on train :{sum(max_epoch_acc_train)/args.runs}")
        print(f"Average accuracy over {args.runs} iterations on valid :{sum(max_epoch_acc_valid)/args.runs}")
        print(f"Highest accuracy over train: {max(max_epoch_acc_train)}")
        print(f"Highest accuracy over valid: {max(max_epoch_acc_valid)}")

    def print_statistics(self, run=None):
        """ Original method, not used """
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

    def save_results(self, args):
        """ saves the results in separate files in a results directory """

        if not os.path.exists('results'):
            os.makedirs('results')

        if args.inductive:
            with open('./results/results_inductive_{}runs'.format(args.runs), 'wb') as output:
                pickle.dump(self.results, output)

        if args.transductive:
            with open('./results/results_transductive_{}runs'.format(args.runs), 'wb') as output:
                pickle.dump(self.results, output)

    def callback_early_stopping(self, valid_accuracies):
        """
        Takes as argument the list with all the validation accuracies.
        If patience=k, checks if the mean of the last k accuracies is higher than the mean of the
        previous k accuracies (i.e. we check that we are not overfitting). If not, stops learning.
        @param valid_accuracies - list(float) , validation accuracy per epoch
        @return bool - if training stops or not

        """
        epoch = len(valid_accuracies)
        # no early stopping for 2 * patience epochs
        if epoch // self.es_patience < 2:
            return False

        # Mean loss for last patience epochs and second-last patience epochs
        mean_previous = np.mean(valid_accuracies[epoch - 2 * self.es_patience:epoch - self.es_patience])
        mean_recent = np.mean(valid_accuracies[epoch - self.es_patience:epoch])
        delta = mean_recent - mean_previous
        if delta <= self.es_min_delta:
            print("*CB_ES* Validation Accuracy didn't increase in the last %d epochs" % self.es_patience)
            print("*CB_ES* delta:", delta)
            print("callback_early_stopping signal received at epoch= %d" % len(valid_accuracies))
            print("Terminating training")
            return True
        else:
            return False
