# train KENN here
# this should later on be done in another file but to keep the overview I have it in a separate file now
# Remark: only transductive training at the moment, only one base NN (= MLP)

import argparse
from time import time

import torch
import torch.nn.functional as F
import torch_geometric
from torch.utils.tensorboard.writer import SummaryWriter

from RangeConstraint import RangeConstraint
from generate_knowledge import generate_knowledge
from logger import Logger
from logger import reset_folders
from model import get_model
from ogb.nodeproppred import Evaluator
from preprocess_data import load_and_preprocess
from training_batch import train, test


def main():
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')  # alternatively products
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)  # 500
    parser.add_argument('--runs', type=int, default=1)  # 10
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--mode', type=str, default='transductive')  # inductive/transductive
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--binary_preactivation', type=float, default=500.0)
    parser.add_argument('--num_kenn_layers', type=int, default=3)
    parser.add_argument('--range_constraint_lower', type=float, default=0)
    parser.add_argument('--range_constraint_upper', type=float, default=500)
    parser.add_argument('--es_enabled', type=bool, default=False)
    parser.add_argument('--es_min_delta', type=float, default=0.001)
    parser.add_argument('--es_patience', type=int, default=3)
    parser.add_argument('--sampling_neighbor_size', type=int, default=-1)  # all neighbors will be included with -1
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--full_batch', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()
    print(args)

    torch_geometric.seed_everything(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Cuda available? {torch.cuda.is_available()}, Number of devices: {torch.cuda.device_count()}')

    if args.mode == 'transductive':

        data, split_idx, train_batches, valid_batches, test_batches = load_and_preprocess(args)
        _ = generate_knowledge(data.num_classes)

        print('Start Transductive Training')

        logger = Logger(args)
        reset_folders(args)
        range_constraint = RangeConstraint(lower=args.range_constraint_lower, upper=args.range_constraint_upper)

        for run in range(args.runs):
            print(f"Run: {run} of {args.runs}")
            writer = SummaryWriter('runs/' + args.dataset + f'/transductive/run{run}')
            model = get_model(data, args).to(device)
            evaluator = Evaluator(name='ogbn-arxiv')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = F.nll_loss

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []

            if model.name.startswith('KENN'):
                clause_weights_dict = {f"clause_weights_{i}": [] for i in range(args.num_kenn_layers)}
            else:
                clause_weights_dict = None

            for epoch in range(args.epochs):
                print(f'Start batch training of epoch {epoch}')
                print(f"Number of Training batches with batch_size = {args.batch_size}: {len(train_batches)}")
                start = time()
                t_loss = train(model, train_batches, optimizer, device, criterion, range_constraint)
                t_accuracy, _ = test(model, train_batches, criterion, device, evaluator)
                v_accuracy, v_loss = test(model, valid_batches, criterion, device, evaluator)
                test_accuracy, _ = test(model, test_batches, criterion, device, evaluator)
                end = time()

                writer.add_scalar("loss/train", t_loss, epoch)
                writer.add_scalar("loss/valid", v_loss, epoch)
                writer.add_scalar("accuracy/train", t_accuracy, epoch)
                writer.add_scalar("accuracy/valid", v_accuracy, epoch)

                train_accuracies.append(t_accuracy)
                valid_accuracies.append(v_accuracy)
                train_losses.append(t_loss)
                valid_losses.append(v_loss)

                if model.name.startswith('KENN'):
                    for i in range(args.num_kenn_layers):
                        clause_weights_dict[f"clause_weights_{i}"].append(
                            [ce.clause_weight for ce in model.kenn_layers[i].binary_ke.clause_enhancers])

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {t_loss:.4f}, '
                          f'Time: {end - start:.6f} '
                          f'Train: {100 * t_accuracy:.2f}%, '
                          f'Valid: {100 * v_accuracy:.2f}% '
                          f'Test: {100 * test_accuracy:.2f}% ')

                # early stopping
                if args.es_enabled and logger.callback_early_stopping(valid_accuracies):
                    break

            test_accuracy = test(model, test_batches, criterion, device, evaluator)
            logger.add_result(train_losses, train_accuracies, valid_losses, valid_accuracies, test_accuracy, run,
                              clause_weights_dict)
            writer.close()

        logger.print_results(args)
        logger.save_results(args)

    if args.mode == 'inductive':

        data, split_idx, train_batches, valid_batches, test_batches = load_and_preprocess(args)
        _ = generate_knowledge(data.num_classes)

        print('Start Inductive Training')
        logger = Logger(args)
        reset_folders(args)
        range_constraint = RangeConstraint(lower=args.range_constraint_lower, upper=args.range_constraint_upper)

        for run in range(args.runs):
            print(f"Run: {run} of {args.runs}")
            writer = SummaryWriter('runs/' + args.dataset + f'/inductive/run{run}')
            model = get_model(data, args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = F.nll_loss

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []

            if model.name.startswith('KENN'):
                clause_weights_dict = {f"clause_weights_{i}": [] for i in range(args.num_kenn_layers)}
            else:
                clause_weights_dict = None

            for epoch in range(args.epochs):
                print(f'Start batch training of epoch {epoch}')
                print(f"Number of Training batches with batch_size = {args.batch_size}: {len(train_batches)}")
                t_accuracy, t_loss = train(model, train_batches, optimizer, device, criterion, range_constraint)
                v_accuracy, v_loss = test(model, valid_batches, criterion, device)

                writer.add_scalar("loss/train", t_loss, epoch)
                writer.add_scalar("loss/valid", v_loss, epoch)
                writer.add_scalar("accuracy/train", t_accuracy, epoch)
                writer.add_scalar("accuracy/valid", v_accuracy, epoch)

                train_accuracies.append(t_accuracy)
                valid_accuracies.append(v_accuracy)
                train_losses.append(t_loss)
                valid_losses.append(v_loss)

                if model.name.startswith('KENN'):
                    for i in range(args.num_kenn_layers):
                        clause_weights_dict[f"clause_weights_{i}"].append(
                            [ce.clause_weight for ce in model.kenn_layers[i].binary_ke.clause_enhancers])

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {t_loss:.4f}, '
                          f'Train: {100 * t_accuracy:.2f}%, '
                          f'Valid: {100 * v_accuracy:.2f}% ')

                # early stopping
                if args.es_enabled and logger.callback_early_stopping(valid_accuracies):
                    break

            test_accuracy = test(model, test_batches, criterion, device)
            logger.add_result(train_losses, train_accuracies, valid_losses, valid_accuracies, test_accuracy, run,
                              clause_weights_dict)
            writer.close()

        logger.print_results(args)
        logger.save_results(args)


if __name__ == '__main__':
    main()
