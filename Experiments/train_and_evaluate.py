from time import time
import torch.nn.functional as F
import torch_geometric
import wandb
from app_stats import RunStats, ExperimentStats
from model import get_model
from evaluate import Evaluator
from preprocess_data import *
from training_batch import train, test
from pathlib import Path
import json
import pickle


class ExperimentConf(object):
    def __init__(self, conf_dict):
        for key, value in conf_dict.items():
            setattr(self, key, value)


def run_experiment(args):
    torch_geometric.seed_everything(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Cuda available? {torch.cuda.is_available()}, Number of devices: {torch.cuda.device_count()}')

    print(f'Start Training')
    results = dict.fromkeys(['test_accuracies'])
    xp_stats = ExperimentStats()
    test_accuracies = []
    evaluator = Evaluator(args)

    for run in range(args.runs):

        print(f"Run: {run} of {args.runs}")
        model = get_model(args).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                                     eps=args.adam_eps, amsgrad=False)
        criterion = F.nll_loss

        train_losses, valid_losses, train_accuracies, valid_accuracies, epoch_time, clause_weights = [], [], [], [], [], []

        for epoch in range(args.epochs):
            start = time()
            evaluator.track_clause_weights(run, model)
            train(model, optimizer, device, criterion)
            end = time()

            if epoch % args.eval_steps == 0:
                _, t_accuracy, v_accuracy, t_loss, v_loss, _ = test(model, criterion, device, evaluator)

                train_accuracies += [t_accuracy]
                valid_accuracies += [v_accuracy]
                train_losses += [t_loss]
                valid_losses += [v_loss]
                epoch_time += [end - start]

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {t_loss:.4f}, '
                      f'Time per Train Step: {end - start:.6f} '
                      f'Train: {100 * t_accuracy:.2f}%, '
                      f'Valid: {100 * v_accuracy:.2f}% ')

            # early stopping
            if args.es_enabled and evaluator.callback_early_stopping(valid_accuracies, epoch):
                print(f'Early Stopping at epoch {epoch}.')
                break

        test_accuracy, valid_acc, *_ = test(model, criterion, device, evaluator)
        test_accuracies += [test_accuracy]
        rs = RunStats(run, train_losses, train_accuracies, valid_losses, valid_accuracies, test_accuracy, epoch_time,
                      test_accuracies)
        xp_stats.add_run(rs)
        print(rs)
        if args.wandb_use:
            wandb.log(rs.to_dict())
            wandb.log({'valid_acc': valid_acc})
            wandb.run.summary["test_accuracies"] = test_accuracies

        # store the results of run
        results['test_accuracies'] = test_accuracies
        pth = Path(__file__).parent / 'results'
        if not pth.exists():
            pth.mkdir()
        results_dir = pth / str('results_' + str(args.dataset) + '_' + str(args.model))
        with open(results_dir, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    evaluator.save_clause_weights()
    xp_stats.end_experiment()
    print(xp_stats)
    if args.wandb_use:
        wandb.log(xp_stats.to_dict())


if __name__ == '__main__':
    # Opening conf file
    path = Path.cwd() / 'conf.json'
    with open(path, 'r') as f:
        json_content = json.loads(f.read())
    # run experiments
    for conf in json_content['configs']:
        run_experiment(ExperimentConf(conf))
