""" This script runs the training/evaluation loops """
"""
At the moment it only works for base NN 
TODO: 
- adapt a script for KENN and add a script that runs everything 
- early stopping ? 
- to discuss: use to_symmetric() for links 
- prepare for inductive and transductive setting -IN PROGRESS 

"""

import os
import argparse
import pickle
from initialize_model import initialize
import torch
import torch_sparse
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger
from model import GCN, SAGE, Standard, MLP
import torch_geometric.transforms as T
from training import *


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=4)  # 500
    parser.add_argument('--runs', type=int, default=3)  # 10
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--inductive', type=bool, default=True)
    parser.add_argument('--transductive', type=bool, default=True)
    parser.add_argument('--save_results', action='store_true')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor(remove_edge_index=False))
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.adj_t = torch_sparse.SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                           sparse_sizes=(data.num_nodes, data.num_nodes)).to_symmetric()
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.inductive:
        mask_train = torch.all(torch.isin(data.edge_index, split_idx['train']), dim=0)
        mask_valid = torch.all(torch.isin(data.edge_index, split_idx['valid']), dim=0)
        mask_test = torch.all(torch.isin(data.edge_index, split_idx['test']), dim=0)

        data.adj_train = torch_sparse.SparseTensor(row=torch.masked_select(input=data.edge_index[0], mask=mask_train),
                                                   col=torch.masked_select(input=data.edge_index[1], mask=mask_train),
                                                   sparse_sizes=(data.num_nodes, data.num_nodes)).to_symmetric()

        data.adj_valid = torch_sparse.SparseTensor(row=torch.masked_select(input=data.edge_index[0], mask=mask_valid),
                                                   col=torch.masked_select(input=data.edge_index[1], mask=mask_valid),
                                                   sparse_sizes=(data.num_nodes, data.num_nodes)).to_symmetric()

        data.adj_test = torch_sparse.SparseTensor(row=torch.masked_select(input=data.edge_index[0], mask=mask_test),
                                                  col=torch.masked_select(input=data.edge_index[1], mask=mask_test),
                                                  sparse_sizes=(data.num_nodes, data.num_nodes)).to_symmetric()

    data = data.to(device)  # todo: does this work ?
    # x = data.x.to(device)

    # INITIALIZE THE MODEL
    model = initialize(args, data)
    model.to(device)

    evaluator = Evaluator(name='ogbn-arxiv')

    """HERE THE TRAINING LOOP STARTS"""
    if args.transductive:
        logger = Logger(model.name)
        print('Start transductive training')

        for run in range(args.runs):
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []

            for epoch in range(args.epochs):
                t_loss = train_transductive(model, data, train_idx, optimizer)
                train_acc, valid_acc, test_acc, out = test_transductive(model, data, split_idx, evaluator)
                v_loss = F.nll_loss(out[split_idx['valid']], data.y.squeeze(1)[split_idx['valid']]).item()

                train_accuracies.append(train_acc)
                valid_accuracies.append(valid_acc)
                train_losses.append(t_loss)
                valid_losses.append(v_loss)

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {t_loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}% '
                          f'Test: {100 * test_acc:.2f}%')

            logger.add_result(train_losses, train_accuracies, valid_losses, valid_accuracies, test_acc, run)
        logger.print_results(args, 'transductive')

        logger.save_results(args)

    if args.inductive:
        logger = Logger(model.name)
        for run in range(args.runs):
            print('Start Inductive Runs ')
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            train_losses = []
            valid_losses = []
            train_accuracies = []
            valid_accuracies = []

            for epoch in range(1, 1 + args.epochs):
                t_loss = train_inductive(model, data, train_idx, optimizer)
                acc, out = test_inductive(model, data, split_idx, evaluator)  # adapt
                v_loss = F.nll_loss(out[1], data.y.squeeze(1)[split_idx['valid']]).item()

                train_accuracies.append(acc[0])
                valid_accuracies.append(acc[1])
                train_losses.append(t_loss)
                valid_losses.append(v_loss)

                if epoch % args.log_steps == 0:
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {t_loss:.4f}, '
                          f'Train: {100 * acc[0]:.2f}%, '
                          f'Valid: {100 * acc[1]:.2f}% '
                          f'Test: {100 * acc[2]:.2f}%')

            logger.add_result(train_losses, train_accuracies, valid_losses, valid_accuracies, acc[2], run)
        logger.save_results(args)
        logger.print_results(args, 'inductive')

if __name__ == "__main__":
    main()
