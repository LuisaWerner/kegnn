# train kenn-sub-Experiments here
# this should later on be done in another file but to keep the overview I have it in a separate file now
# Remark: only transductive training at the moment, only one base NN (= MLP)
import argparse
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch.utils.tensorboard.writer import SummaryWriter

import wandb
from app_stats import RunStats, ExperimentStats
from generate_knowledge import generate_knowledge
from logger import reset_folders
from model import get_model
from ogb.nodeproppred import Evaluator
from preprocess_data import load_and_preprocess
from training_batch import train, test


def callback_early_stopping(valid_accuracies, es_patience, es_min_delta):
    """
    Takes as argument the list with all the validation accuracies.
    If patience=k, checks if the mean of the last k accuracies is higher than the mean of the
    previous k accuracies (i.e. we check that we are not overfitting). If not, stops learning.
    @param valid_accuracies - list(float) , validation accuracy per epoch
    @param es_patience: early stopping patience
    @param es_min_delta: early stopping delta. Minimum threshold above which the model is considered improving.
    @return bool - if training stops or not

    """
    epoch = len(valid_accuracies)

    # no early stopping for 2 * patience epochs
    if epoch // es_patience < 2:
        return False

    # Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(valid_accuracies[epoch - 2 * es_patience:epoch - es_patience])
    mean_recent = np.mean(valid_accuracies[epoch - es_patience:epoch])
    delta = mean_recent - mean_previous
    if delta <= es_min_delta:
        print("*CB_ES* Validation Accuracy didn't increase in the last %d epochs" % es_patience)
        print("*CB_ES* delta:", delta)
        print("callback_early_stopping signal received at epoch= %d" % len(valid_accuracies))
        print("Terminating training")
        return True
    else:
        return False


def run_experiment(args):
    torch_geometric.seed_everything(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Cuda available? {torch.cuda.is_available()}, Number of devices: {torch.cuda.device_count()}')

    print(f'Start {args.mode} Training')
    reset_folders(args)
    xp_stats = ExperimentStats()

    for run in range(args.runs):

        # load data todo: do this outside or inside the runs loop ?
        data, train_loader, all_loader = load_and_preprocess(
            args)  # todo is this supposed to be outside of the loop? I dont think so
        _ = generate_knowledge(data.num_classes)

        print(f"Run: {run} of {args.runs}")
        print(f"Number of Training Batches with batch_size = {args.batch_size}: {len(train_loader)}")
        writer = SummaryWriter('runs/' + args.dataset + f'/{args.mode}/run{run}')

        model = get_model(data, args).to(device)
        evaluator = Evaluator(name=args.dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = F.nll_loss

        train_losses = []
        valid_losses = []
        train_accuracies = []
        valid_accuracies = []
        epoch_time = []

        clause_weights_dict = None

        for epoch in range(args.epochs):
            start = time()
            t_loss = train(model, train_loader, optimizer, device, criterion, args)
            t_accuracy, v_accuracy, _, _, v_loss, _ = test(model, all_loader, criterion, device, evaluator, data)
            end = time()

            # Save stats for tensorboard
            writer.add_scalar("loss/train", t_loss, epoch)
            writer.add_scalar("loss/valid", v_loss, epoch)
            writer.add_scalar("accuracy/train", t_accuracy, epoch)
            writer.add_scalar("accuracy/valid", v_accuracy, epoch)

            train_accuracies.append(t_accuracy)
            valid_accuracies.append(v_accuracy)
            train_losses.append(t_loss)
            valid_losses.append(v_loss)
            epoch_time.append(end - start)


            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {t_loss:.4f}, '
                      f'Time: {end - start:.6f} '
                      f'Train: {100 * t_accuracy:.2f}%, '
                      f'Valid: {100 * v_accuracy:.2f}% ')

            # early stopping
            if args.es_enabled and callback_early_stopping(valid_accuracies):
                print(f'Early Stopping at epoch {epoch}.')
                break

        # test_accuracy = test(model, test_batches, criterion, device, evaluator)
        _, _, test_accuracy, _, _, _ = test(model, all_loader, criterion, device, evaluator, data)
        rs = RunStats(run, train_losses, train_accuracies, valid_losses, valid_accuracies, test_accuracy, epoch_time)
        xp_stats.add_run(rs)
        print(rs)
        wandb.log(rs.to_dict())
        writer.close()

        xp_stats.end_experiment()
        print(xp_stats)
        wandb.log(xp_stats.to_dict())


def main():
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')  # alternatively products
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)  # todo
    parser.add_argument('--num_layers_sampling', type=int,
                        default=1)  # has to correspond to the number of kenn-sub/GCN Layers
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)  # 500
    parser.add_argument('--runs', type=int, default=1)  # 10
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--mode', type=str, default='transductive',
                        help='transductive or inductive training mode ')  # inductive/transductive
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--binary_preactivation', type=float, default=500.0)
    parser.add_argument('--num_kenn_layers', type=int, default=3)
    parser.add_argument('--range_constraint_lower', type=float, default=0)
    parser.add_argument('--range_constraint_upper', type=float, default=500)
    parser.add_argument('--es_enabled', type=bool, default=False)
    parser.add_argument('--es_min_delta', type=float, default=0.001)
    parser.add_argument('--es_patience', type=int, default=3)
    parser.add_argument('--sampling_neighbor_size', type=int, default=-1)  # all neighbors will be included with -1
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--full_batch', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--train_sampling', type=str, default='graph_saint',
                        help='specify as "cluster", "graph_saint". If '
                             'not specified, standard GraphSAGE sampling '
                             'is applied')
    parser.add_argument('--cluster_sampling_num_partitions', type=int, default=15,
                        help='argument for cluster sampling: In how many partitions should the graph be clustered.')
    parser.add_argument('--sample_coverage', type=int, default=0, help='argument for graph saint, if sample coverage '
                                                                       'is 0, no normalization of batches is '
                                                                       'conducted ')
    parser.add_argument('--walk_length', type=int, default=3, help='argument for graph saint')
    parser.add_argument('--num_steps', type=int, default=30, help='argument for graph saint')

    args = parser.parse_args()
    print(args)

    run_experiment(args)


if __name__ == '__main__':
    main()
