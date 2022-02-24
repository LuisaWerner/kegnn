# train KENN here
# this should later on be done in another file but to keep the overview I have it in a separate file now
# Remark: only transductive training at the moment, only one base NN (= MLP)

import argparse

import torch_geometric
from torch.utils.tensorboard.writer import SummaryWriter

from RangeConstraint import RangeConstraint
from generate_knowledge import generate_knowledge
from logger import Logger
from logger import reset_folders
from model import KENN
from ogb.nodeproppred import Evaluator
from preprocess_data import load_and_preprocess
from training import *
from training_batch import train, test


def main():
    parser = argparse.ArgumentParser(description='Experiments - KENN with MLP ')
    parser.add_argument('--dataset', type=str, default='ogbn-products')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')  # todo no effect
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=2)  # 500
    parser.add_argument('--runs', type=int, default=2)  # 10
    parser.add_argument('--model', type=str, default='MLP')  # todo : make dependent from args
    parser.add_argument('--mode', type=str, default='inductive')  # alternatively inductive
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--binary_preactivation', type=float, default=500.0)
    parser.add_argument('--num_kenn_layers', type=int, default=3)
    parser.add_argument('--range_constraint_lower', type=float, default=0)
    parser.add_argument('--range_constraint_upper', type=float, default=500)
    parser.add_argument('--es_min_delta', type=float, default=0.001)
    parser.add_argument('--es_patience', type=int, default=3)
    parser.add_argument('--sampling_neighbor_size', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--full_batch', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()
    print(args)

    torch_geometric.seed_everything(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.mode == 'transductive':

        data, split_idx, train_batches, valid_batches, test_batches = load_and_preprocess(args)

        # INITIALIZE THE MODEL
        evaluator = Evaluator(name=args.dataset)
        _ = generate_knowledge(data.num_classes)

        print('Start Transductive Training')
        model = KENN(knowledge_file='knowledge_base',
                     in_channels=data.num_features,
                     out_channels=data.num_classes,
                     hidden_channels=args.hidden_channels,
                     num_layers=args.num_layers,
                     num_kenn_layers=args.num_kenn_layers,
                     dropout=args.dropout)

        model.to(device)
        logger = Logger(model.name, args)
        reset_folders(args)
        range_constraint = RangeConstraint(lower=args.range_constraint_lower, upper=args.range_constraint_upper)

        for run in range(args.runs):
            print(f"Run: {run} of {args.runs}")
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = F.nll_loss

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []

            clause_weights_dict = {f"clause_weights_{i}": [] for i in range(args.num_kenn_layers)}

            writer = SummaryWriter('runs/' + args.dataset + f'/transductive/run{run}')
            for epoch in range(args.epochs):
                print(f'Start batch training of epoch {epoch}')
                print(f"Number of Training batches with batch_size = {args.batch_size}: {len(train_batches)}")
                t_accuracy, t_loss = train(model, train_batches, optimizer, device, criterion, args, range_constraint)
                v_accuracy, v_loss = test(model, valid_batches, criterion, args, device)

                writer.add_scalar("loss/train", t_loss, epoch)
                writer.add_scalar("loss/valid", v_loss, epoch)
                writer.add_scalar("accuracy/train", t_accuracy, epoch)
                writer.add_scalar("accuracy/valid", v_accuracy, epoch)

                train_accuracies.append(t_accuracy)
                valid_accuracies.append(v_accuracy)
                train_losses.append(t_loss)
                valid_losses.append(v_loss)

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
                if logger.callback_early_stopping(valid_accuracies):
                    break

            test_accuracy = test(model, test_batches, criterion, args, device)
            logger.add_result(train_losses, train_accuracies, valid_losses, valid_accuracies, test_accuracy, run,
                              clause_weights_dict)
            # writer.flush()
            writer.close()

        logger.print_results(args)
        logger.save_results(args)

    if args.mode == 'inductive':

        data, split_idx, train_batches, valid_batches, test_batches = load_and_preprocess(args)

        # INITIALIZE THE MODEL
        evaluator = Evaluator(name=args.dataset)
        _ = generate_knowledge(data.num_classes)

        print('Start Inductive Training')
        model = KENN(knowledge_file='knowledge_base',
                     in_channels=data.num_features,
                     out_channels=data.num_classes,
                     hidden_channels=args.hidden_channels,
                     num_layers=args.num_layers,
                     num_kenn_layers=args.num_kenn_layers,
                     dropout=args.dropout)

        model.to(device)
        logger = Logger(model.name, args)
        reset_folders(args)
        range_constraint = RangeConstraint(lower=args.range_constraint_lower, upper=args.range_constraint_upper)

        for run in range(args.runs):
            print(f"Run: {run} of {args.runs}")
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = F.nll_loss

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []

            clause_weights_dict = {f"clause_weights_{i}": [] for i in range(args.num_kenn_layers)}

            writer = SummaryWriter('runs/' + args.dataset + f'/inductive/run{run}')
            for epoch in range(args.epochs):
                print(f'Start batch training of epoch {epoch}')
                print(f"Number of Training batches with batch_size = {args.batch_size}: {len(train_batches)}")
                t_accuracy, t_loss = train(model, train_batches, optimizer, device, criterion, args, range_constraint)
                v_accuracy, v_loss = test(model, valid_batches, criterion, args, device)

                writer.add_scalar("loss/train", t_loss, epoch)
                writer.add_scalar("loss/valid", v_loss, epoch)
                writer.add_scalar("accuracy/train", t_accuracy, epoch)
                writer.add_scalar("accuracy/valid", v_accuracy, epoch)

                train_accuracies.append(t_accuracy)
                valid_accuracies.append(v_accuracy)
                train_losses.append(t_loss)
                valid_losses.append(v_loss)

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
                if logger.callback_early_stopping(valid_accuracies):
                    break

            test_accuracy = test(model, test_batches, criterion, args, device)
            logger.add_result(train_losses, train_accuracies, valid_losses, valid_accuracies, test_accuracy, run,
                              clause_weights_dict)
            # writer.flush()
            writer.close()

        logger.print_results(args)
        logger.save_results(args)


if __name__ == '__main__':
    main()
