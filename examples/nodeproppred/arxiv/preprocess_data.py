import torch
import torch_sparse
from torch_geometric.loader import NeighborLoader

from ogb.nodeproppred import PygNodePropPredDataset


def get_inductive(subset, args):
    """
    replaces the full adjacency matrix by a reduced adjacency matrix with inductive links

    """
    dataset = PygNodePropPredDataset(name=args.dataset)
    data = dataset[0]
    data.num_classes = dataset.num_classes

    split_idx = dataset.get_idx_split()

    mask = torch.all(torch.isin(data.edge_index, split_idx[subset]), dim=0)
    adj_t = torch_sparse.SparseTensor(row=torch.masked_select(input=data.edge_index[0], mask=mask),
                                      col=torch.masked_select(input=data.edge_index[1], mask=mask),
                                      sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = adj_t
    data.relations = torch.full(size=(torch.count_nonzero(mask).item(), 1),
                                fill_value=args.binary_preactivation)
    return data, split_idx


def load_and_preprocess(args):
    """
    loads data and create batches
    @param args: Argument Parser object specified by user
    @return train_loader, valid_loader, test_loader: train/valid/test set split into batches,
    samples neighbors specified in args in an transductive
    #todo for inductive remove nodes from data before
    """

    if args.mode == 'transductive':
        dataset = PygNodePropPredDataset(name=args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.adj_t = torch_sparse.SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                               sparse_sizes=(data.num_nodes, data.num_nodes))
        data.relations = torch.full(size=(data.num_edges, 1), fill_value=args.binary_preactivation)

        split_idx = dataset.get_idx_split()

        if args.batch_size > data.num_nodes:
            print('choose batch size smaller than the source dataset to create batches from ')
            args.batch_size = data.num_nodes

        train_loader_transductive = NeighborLoader(data,
                                                   num_neighbors=[args.sampling_neighbor_size] * 2,
                                                   shuffle=True,
                                                   input_nodes=split_idx['train'],
                                                   batch_size=args.batch_size)

        valid_loader_transductive = NeighborLoader(data,
                                                   num_neighbors=[args.sampling_neighbor_size],
                                                   shuffle=True,
                                                   input_nodes=split_idx['valid'],
                                                   batch_size=args.batch_size)
        test_loader_transductive = NeighborLoader(data,
                                                  num_neighbors=[args.sampling_neighbor_size],
                                                  shuffle=True,
                                                  input_nodes=split_idx['test'],
                                                  batch_size=args.batch_size)

        return data, split_idx, train_loader_transductive, valid_loader_transductive, test_loader_transductive

    elif args.mode == 'inductive':
        data_train, split_idx = get_inductive('train', args)
        data_valid, _ = get_inductive('valid', args)
        data_test, _ = get_inductive('test', args)

        train_loader_inductive = NeighborLoader(data=data_train,
                                                num_neighbors=[args.sampling_neighbor_size],
                                                shuffle=True,
                                                input_nodes=split_idx['train'],
                                                batch_size=args.batch_size)
        valid_loader_inductive = NeighborLoader(data=data_valid,
                                                num_neighbors=[args.sampling_neighbor_size],
                                                shuffle=True,
                                                input_nodes=split_idx['valid'],
                                                batch_size=args.batch_size)
        test_loader_inductive = NeighborLoader(data=data_test,
                                               num_neighbors=[args.sampling_neighbor_size],
                                               shuffle=True,
                                               input_nodes=split_idx['test'],
                                               batch_size=args.batch_size)

        return data_train, split_idx, train_loader_inductive, valid_loader_inductive, test_loader_inductive
