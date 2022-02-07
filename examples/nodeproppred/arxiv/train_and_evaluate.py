""" This script runs the training/evaluation loops """
"""
At the moment it only works for base NN 
TODO: 
- adapt a script for KENN and add a script that runs everything 
- early stopping ? 
- to discuss: use to_symmetric() for links 
- prepare for inductive and transductive setting -IN PROGRESS 

"""

import argparse
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
    parser.add_argument('--epochs', type=int, default=3)  # 500
    parser.add_argument('--runs', type=int, default=1)  # 10
    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--inductive', type=bool, default=True)
    parser.add_argument('--transductive', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor(remove_edge_index=False))
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
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

    data = data.to(device)

    # INITIALIZE THE MODEL
    if args.model == 'MLP':
        if args.use_node_embedding:
            embedding = torch.load('embedding.pt', map_location='cpu')
            data.x = torch.cat([data.x, embedding], dim=-1)
        model = MLP(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                    args.num_layers, args.dropout).to(device)

    elif args.model == 'Standard':
        # set parameters for Standard NN of KENN
        args.use_node_embedding = False
        args.num_layers = 4
        args.hidden_channels = 50
        args.dropout = 0.5
        args.lr = 0.01
        args.epochs = 300
        args.runs = 500

        if args.use_node_embedding:
            embedding = torch.load('embedding.pt', map_location='cpu')
            data.x = torch.cat([data.x, embedding], dim=-1)

        model = Standard(data.x.size(-1), args.hidden_channels, dataset.num_classes,
                         args.num_layers, args.dropout).to(device)

    elif args.model == 'SAGE':
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)

    elif args.model == 'GCN':
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
    else:
        print('unknown model :', args.model)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    """HERE THE TRAINING LOOP STARTS"""
    if args.transductive:
        print('Start transductive training')
        for run in range(args.runs):
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(1, 1 + args.epochs):
                # todo: adapt for inductive case
                loss = train_transductive(model, data, train_idx, optimizer)
                result = test_transductive(model, data, split_idx, evaluator)
                logger.add_result(run, result) #adapt

                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}% '
                          f'Test: {100 * test_acc:.2f}%')

            logger.print_statistics(run)  # adapt
        logger.print_statistics() #adapt

    if args.inductive:
        for run in range(args.runs):
            print('Start Inductive Runs ')
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(1, 1 + args.epochs):
                loss = train_inductive(model, data, train_idx, optimizer) # adapt
                result = test_inductive(model, data, split_idx, evaluator) # adapt
                logger.add_result(run, result) # adapt

                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}% '
                          f'Test: {100 * test_acc:.2f}%')

            logger.print_statistics(run) #adapt
        logger.print_statistics() #adapt




if __name__ == "__main__":
    main()
