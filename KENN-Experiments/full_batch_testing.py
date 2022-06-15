import argparse
from time import time

import torch
import torch.nn.functional as F
import torch_geometric

from model import get_model
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


def train(model, data, train_idx, optimizer, criterion):
    """
    training step for transductive setting
    @param model: callable NN model of torch.nn.Module
    @param data: data object with x, y, adjacency matrix (full graph)
    @param optimizer: torch.optim object
    @param train_idx: training split
    @param range_constraint object of RangeConstraint.py to constrain parameters
    @returns: loss (float)
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.relations)[train_idx]
    loss = criterion(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    """
    test_transductive
    @param model - should be a NN of type torch.nn.module
    @param data - a PyG data object with x, y, adjacency matrix (all links in full graph)
    @param split_idx - dictionary for split into train/valid/test
    @param evaluator - an evaluator object
    return: accuracy (float) on train, valid, test set
    """
    model.eval()

    out = model(data.x, data.edge_index, data.relations)
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

    return train_acc, valid_acc, test_acc, out


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
    parser.add_argument('--epochs', type=int, default=500)  # 500
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
    parser.add_argument('--full_batch', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=100)

    args = parser.parse_args()
    print(args)

    torch_geometric.seed_everything(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Cuda available? {torch.cuda.is_available()}, Number of devices: {torch.cuda.device_count()}')

    # dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
    dataset = PygNodePropPredDataset(name='ogbn-arxiv')

    data = dataset[0]
    # data.adj_t = data.adj_t.to_symmetric()
    data.num_classes = dataset.num_classes
    data.relations = torch.full(size=(data.num_edges, 1), fill_value=args.binary_preactivation)
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    if args.mode == 'transductive':

        for run in range(args.runs):
            print(f"Run: {run} of {args.runs}")
            model = get_model(data, args).to(device)
            evaluator = Evaluator(name='ogbn-arxiv')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = F.nll_loss

            for epoch in range(args.epochs):
                print(f'Start batch training of epoch {epoch}')
                start = time()
                loss = train(model, data, train_idx, optimizer, criterion)
                train_acc, valid_acc, test_acc, out = test(model, data, split_idx, evaluator)
                end = time()

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Time: {end - start:.6f} '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')


if __name__ == "__main__":
    main()
