""" This script runs the training/evaluation loops """
"""
At the moment it only works for base NN 
TODO: 
- adapt a script for KENN and add a script that runs everything 
- early stopping ? 
- prepare for inductive and transductive setting 
"""

import argparse
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from logger import Logger
from model import GCN, SAGE, Standard, MLP
import torch_geometric.transforms as T

def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

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
    parser.add_argument('--epochs', type=int, default=3) # 500
    parser.add_argument('--runs', type=int, default=1) # 10
    parser.add_argument('--model', type=str, default='MLP')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

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

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()

if __name__ == "__main__":
    main()