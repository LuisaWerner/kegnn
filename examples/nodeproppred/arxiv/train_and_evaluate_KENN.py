# train KENN here
# this should later on be done in another file but to keep the overview I have it in a separate file now
# Remark: only transductive training at the moment, only one base NN (= MLP)

import os
import argparse
import pickle
from initialize_model import initialize
import torch
import torch_sparse
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger
from model import GCN, SAGE, Standard, MLP, KENN
import torch_geometric.transforms as T
from training import *
from generate_knowledge import generate_knowledge

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
    parser.add_argument('--binary_preactivation', type=float, default=500.0)
    parser.add_argument('--num_kenn_layers', type=int, default=3)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.adj_t = torch_sparse.SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                           sparse_sizes=(data.num_nodes, data.num_nodes)).to_symmetric()
    data.relations = torch.full(size=(data.num_edges, 1), fill_value=args.binary_preactivation)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device) #todo: needed?

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


        data.relations_train = torch.full(size=(torch.count_nonzero(mask_train).item(), 1), fill_value=args.binary_preactivation) #todo verify shape
        data.relations_valid = torch.full(size=(torch.count_nonzero(mask_valid).item(), 1), fill_value=args.binary_preactivation) #todo verify shape
        data.relations_test = torch.full(size=(torch.count_nonzero(mask_test).item(), 1), fill_value=args.binary_preactivation) #todo verify shape
    data = data.to(device)

    # INITIALIZE THE MODEL
    # model = initialize(args, data)

    evaluator = Evaluator(name='ogbn-arxiv')
    _ = generate_knowledge(data.num_classes)
    if args.transductive:
        model = KENN(knowledge_file='knowledge_base',
                     in_channels=data.num_features,
                     out_channels=data.num_classes,
                     hidden_channels=args.hidden_channels,
                     num_layers=args.num_layers,
                     num_kenn_layers=args.num_kenn_layers,
                     dropout=args.dropout,
                     relations=data.relations) # shape =  (# relations, 1)

        model.to(device)
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

if __name__ == '__main__':
    main()