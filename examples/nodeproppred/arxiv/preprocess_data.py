import torch
import torch_sparse
from torch_geometric.loader import NeighborLoader

from ogb.nodeproppred import PygNodePropPredDataset


def load_and_preprocess(args):
    """
    loads data and create batches
    @param args: Argument Parser object specified by user
    @return train_loader, valid_loader, test_loader: train/valid/test set split into batches,
    samples neighbors specified in args in an inductive setting
    #todo: specify for inductive setting
    """

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

    train_loader_inductive = NeighborLoader(data,
                                            num_neighbors=[args.sampling_neighbor_size] * 2,
                                            shuffle=True,
                                            input_nodes=split_idx['train'],
                                            # since we only sample from nodes of the training set, it's inductive!
                                            batch_size=args.batch_size)  # todo:

    valid_loader_inductive = NeighborLoader(data,
                                            num_neighbors=[args.sampling_neighbor_size],
                                            shuffle=True,
                                            input_nodes=split_idx['valid'],
                                            batch_size=args.batch_size)
    test_loader_inductive = NeighborLoader(data,
                                           num_neighbors=[args.sampling_neighbor_size],
                                           shuffle=True,
                                           input_nodes=split_idx['test'],
                                           batch_size=args.batch_size)

    train_loader_transductive = None  # todo
    valid_loader_transductive = None  # todo
    test_loader_transductive = None  # todo

    return data, split_idx, train_loader_inductive, valid_loader_inductive, test_loader_inductive, train_loader_transductive, valid_loader_transductive, test_loader_transductive
