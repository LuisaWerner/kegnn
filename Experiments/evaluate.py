import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb
import pickle


class Evaluator:
    # _meta_path = Path(Path(__file__).parent.parent)

    def __init__(self, args):
        self.name = args.dataset
        self._meta_path = Path(Path(__file__).parent)
        self.update_meta_info()
        self.meta_info = pd.read_csv(self._meta_path / 'master_all.csv', index_col=0)

        self.es_patience = args.es_patience
        self.eval_steps = args.eval_steps
        self.es_min_delta = args.es_min_delta

        self.best_val_acc = 0.0
        self.name_baseNN = args.model if not args.model.startswith('KENN') else args.model.split('_', 1)[-1]
        self.state_dir = Path.cwd() / 'pretrained_models' / self.name_baseNN / args.dataset
        if not self.state_dir.exists():
            self.state_dir.mkdir(parents=True)

        if self.name not in self.meta_info:
            print(self.name)
            error_msg = f'Invalid dataset name {self.name}.\n'
            error_msg += 'Available datasets are as follows:\n'
            error_msg += '\n'.join(self.meta_info.keys())
            raise ValueError(error_msg)

        self.num_tasks = int(self.meta_info[self.name]['num tasks'])
        self.eval_metric = self.meta_info[self.name]['eval metric']

        self.num_kenn_layers = args.num_kenn_layers
        self.runs = args.runs
        self.dataset = args.dataset
        self.clause_weight_dict = {run_key: {layer_key: {} for layer_key in range(self.num_kenn_layers)}
                                   for run_key in range(self.runs)}

    def track_clause_weights(self, run, model):
        """ tracks clause weights """
        # parameters are named like: kenn_layers.0.binary_ke.clause-0.conorm_boost.clause_weight'
        for name, value in model.named_parameters():
            if 'clause_weight' in name:
                splitted_name = name.split('.')
                clause_number = splitted_name[3].split('-')[1]
                try:
                    self.clause_weight_dict[run][int(splitted_name[1])][clause_number].append(value.item())
                except KeyError:
                    self.clause_weight_dict[run][int(splitted_name[1])].update({clause_number: []})
                    self.clause_weight_dict[run][int(splitted_name[1])][clause_number].append(value.item())

    def save_clause_weights(self):
        with open(f'{self.dataset}_clause_weight_dict', 'wb') as clause_weights:
            pickle.dump(self.clause_weight_dict, clause_weights, protocol=pickle.HIGHEST_PROTOCOL)
        wandb.log({'logged_clause_weights': str(self.clause_weight_dict)})

    def callback_early_stopping(self, valid_accuracies, epoch):
        """
        Takes as argument the list with all the validation accuracies.
        If patience=k, checks if the mean of the last k accuracies is higher than the mean of the
        previous k accuracies (i.e. we check that we are not overfitting). If not, stops learning.
        @param valid_accuracies - list(float) , validation accuracy per epoch
        @param epoch: current epoch
        @param args: argument file [Namespace]
        @return bool - if training stops or not
        """
        step = len(valid_accuracies)
        patience = self.es_patience // self.eval_steps
        # no early stopping for 2 * patience epochs
        if epoch < 2 * self.es_patience:
            return False

        # Mean loss for last patience epochs and second-last patience epochs
        mean_previous = np.mean(valid_accuracies[step - 2 * patience:step - patience])
        mean_recent = np.mean(valid_accuracies[step - patience:step])
        delta = mean_recent - mean_previous
        if delta <= self.es_min_delta:
            print("*CB_ES* Validation Accuracy didn't increase in the last %d epochs" % patience)
            print("*CB_ES* delta:", delta)
            print(f"callback_early_stopping signal received at epoch {epoch}")
            print("Terminating training")
            return True
        else:
            return False

    def update_meta_info(self):
        """
        completes and stores meta info document for all datasets beyond OGB so that a evaluator can be instantiated
        according to the defined metrics here
        """
        ogb_meta_info = pd.read_csv(self._meta_path / 'master.csv', index_col=0)
        datasets = ['CiteSeer', 'Cora', 'PubMed', 'Reddit2', 'AmazonProducts', 'Yelp', 'Flickr']
        for name in datasets:
            meta_info = pd.DataFrame(columns=[name],
                                     index=['num tasks', 'eval metric', 'task type', 'has node attr', 'has edge attr',
                                            'additional node files', 'additional edge files', 'is hetero', 'is binary'])
            meta_info[name]['num tasks'] = 1
            meta_info[name]['eval metric'] = 'acc'
            meta_info[name]['task type'] = 'multiclass classification'
            meta_info[name]['has node attr'] = True
            meta_info[name]['has edge attr'] = False
            meta_info[name]['additional node files'] = None
            meta_info[name]['additional edge files'] = None
            meta_info[name]['is hetero'] = False
            meta_info[name]['is binary'] = False

            ogb_meta_info = pd.concat([ogb_meta_info, meta_info], axis=1)
        ogb_meta_info.to_csv(self._meta_path / 'master_all.csv')

    def plot_grad_flow(self, model, epoch):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in model.named_parameters():
            if p.requires_grad: # and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title(f"Gradient flow: Epoch {epoch}")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.show()

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'acc':
            if 'y_true' not in input_dict:
                raise RuntimeError('Missing key of y_true')
            if 'y_pred' not in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_node, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_node, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred.detach().cpu().numpy()

            # check type
            if not (isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor)):
                y_true, y_pred = torch.Tensor(y_true), torch.Tensor(y_pred)
                # raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred must to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks,
                                                                                             y_true.shape[1]))

            return y_true, y_pred

        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

    def eval(self, input_dictionary):
        """ Evaluation function, returns eval_metric"""
        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dictionary)
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dictionary)
            return self._eval_acc(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

    @property
    def expected_input_format(self):
        desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += 'where y_pred stores score values (for computing ROC-AUC),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        elif self.eval_metric == 'acc':
            desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
            desc += '- y_true: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += '- y_pred: numpy ndarray or torch tensor of shape (num_node, num_task)\n'
            desc += 'where y_pred stores predicted class label (integer),\n'
            desc += 'num_task is {}, and '.format(self.num_tasks)
            desc += 'each row corresponds to one node.\n'
        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

        return desc

    @property
    def expected_output_format(self):
        desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
        if self.eval_metric == 'rocauc':
            desc += '{\'rocauc\': rocauc}\n'
            desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(self.num_tasks)
        elif self.eval_metric == 'acc':
            desc += '{\'acc\': acc}\n'
            desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(self.num_tasks)
        else:
            raise ValueError('Undefined eval metric %s ' % self.eval_metric)

        return desc

    def _eval_rocauc(self, y_true, y_pred):
        '''compute ROC-AUC and AP score averaged across tasks'''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_labeled = y_true[:, i] == y_true[:, i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return {'rocauc': sum(rocauc_list) / len(rocauc_list)}

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list += [float(torch.sum(correct)) / len(correct)]

        return {'acc': sum(acc_list) / len(acc_list)}


if __name__ == '__main__':
    evaluator = Evaluator('ogbn-proteins')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = torch.tensor(np.random.randint(2, size=(100, 112)))
    y_pred = torch.tensor(np.random.randn(100, 112))
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)

    ### acc case
    evaluator = Evaluator('ogbn-products')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = np.random.randint(5, size=(100, 1))
    y_pred = np.random.randint(5, size=(100, 1))
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)

    ### acc case
    evaluator = Evaluator('ogbn-arxiv')
    print(evaluator.expected_input_format)
    print(evaluator.expected_output_format)
    y_true = np.random.randint(5, size=(100, 1))
    y_pred = np.random.randint(5, size=(100, 1))
    input_dict = {'y_true': y_true, 'y_pred': y_pred}
    result = evaluator.eval(input_dict)
    print(result)
