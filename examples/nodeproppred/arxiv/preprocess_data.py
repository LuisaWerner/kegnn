import torch
import torch_sparse
from torch_geometric.loader import NeighborLoader

from ogb.nodeproppred import PygNodePropPredDataset


def load_and_preprocess(args):
    """
    loads data and create batches
    @param args: Argument Parser object specified by user
    @return train_loader, valid_loader, test_loader: train/valid/test set split into batches,
    samples neighbors specified in args in an transductive
    """

    if args.mode == 'transductive':
        dataset = PygNodePropPredDataset(name=args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.adj_t = torch_sparse.SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                               sparse_sizes=(data.num_nodes, data.num_nodes))
        data.relations = torch.full(size=(data.num_edges, 1), fill_value=args.binary_preactivation)
        split_idx = dataset.get_idx_split()

        if args.batch_size > data.num_nodes or args.full_batch:
            print('choose batch size smaller than the source dataset to create batches from ')
            print('Full batch training ')
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
        dataset = PygNodePropPredDataset(name=args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.relations = torch.full(size=(data.num_edges, 1), fill_value=args.binary_preactivation)
        split_idx = dataset.get_idx_split()

        if args.batch_size > data.num_nodes or args.full_batch:
            print('choose batch size smaller than the source dataset to create batches from ')
            args.batch_size = data.num_nodes

        data_train = data.subgraph(split_idx['train'])
        data_valid = data.subgraph(split_idx['valid'])
        data_test = data.subgraph(split_idx['test'])
        data_train.adj_t = torch_sparse.SparseTensor(row=data_train.edge_index[0], col=data_train.edge_index[1],
                                                     sparse_sizes=(data.num_nodes, data.num_nodes))
        data_valid.adj_t = torch_sparse.SparseTensor(row=data_valid.edge_index[0], col=data_valid.edge_index[1],
                                                     sparse_sizes=(data.num_nodes, data.num_nodes))
        data_test.adj_t = torch_sparse.SparseTensor(row=data_test.edge_index[0], col=data_test.edge_index[1],
                                                    sparse_sizes=(data.num_nodes, data.num_nodes))

        train_loader_inductive = NeighborLoader(data=data_train,
                                                num_neighbors=[args.sampling_neighbor_size],
                                                shuffle=True,
                                                input_nodes=None,
                                                batch_size=args.batch_size)
        valid_loader_inductive = NeighborLoader(data=data_valid,
                                                num_neighbors=[args.sampling_neighbor_size],
                                                shuffle=True,
                                                input_nodes=None,
                                                batch_size=args.batch_size)
        test_loader_inductive = NeighborLoader(data=data_test,
                                               num_neighbors=[args.sampling_neighbor_size],
                                               shuffle=True,
                                               input_nodes=None,
                                               batch_size=args.batch_size)

        data.adj_t = torch_sparse.SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                               sparse_sizes=(data.num_nodes, data.num_nodes))

        return data, split_idx, train_loader_inductive, valid_loader_inductive, test_loader_inductive
