import torch
from torch_geometric.loader import NeighborLoader

from ogb.nodeproppred import PygNodePropPredDataset


def load_and_preprocess(args):
    """
    loads data and create batches
    In the inductive setting, the data object is reduced before to avoid links between
    different subsets. Only neighbors from the respective subset are sampled
    @param args: Argument Parser object specified by user
    @return train_loader, valid_loader, test_loader: train/valid/test set split into batches,
    samples neighbors specified in args in an transductive
    """

    if args.mode == 'transductive':
        dataset = PygNodePropPredDataset(name=args.dataset)
        data = dataset[0]
        data.num_classes = dataset.num_classes
        data.relations = torch.full(size=(data.num_edges, 1), fill_value=args.binary_preactivation)
        split_idx = dataset.get_idx_split()

        if args.batch_size > data.num_nodes or args.full_batch:
            print('choose batch size smaller than the source dataset to create batches from ')
            print('Full batch training ')
            args.batch_size = data.num_nodes

        train_loader_transductive = NeighborLoader(data,
                                                   num_neighbors=[args.sampling_neighbor_size] * args.num_layers,
                                                   shuffle=True,
                                                   input_nodes=split_idx['train'],
                                                   batch_size=args.batch_size,
                                                   num_workers=3)

        valid_loader_transductive = NeighborLoader(data,
                                                   num_neighbors=[args.sampling_neighbor_size] * args.num_layers,
                                                   shuffle=True,
                                                   input_nodes=split_idx['valid'],
                                                   batch_size=args.batch_size)

        test_loader_transductive = NeighborLoader(data,
                                                  num_neighbors=[args.sampling_neighbor_size] * args.num_layers,
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

        train_loader_inductive = NeighborLoader(data=data_train,
                                                num_neighbors=[args.sampling_neighbor_size] * args.num_layers,
                                                shuffle=True,
                                                input_nodes=None,
                                                batch_size=args.batch_size,
                                                num_workers=3)
        valid_loader_inductive = NeighborLoader(data=data_valid,
                                                num_neighbors=[args.sampling_neighbor_size] * args.num_layers,
                                                shuffle=True,
                                                input_nodes=None,
                                                batch_size=args.batch_size)
        test_loader_inductive = NeighborLoader(data=data_test,
                                               num_neighbors=[args.sampling_neighbor_size] * args.num_layers,
                                               shuffle=True,
                                               input_nodes=None,
                                               batch_size=args.batch_size)

        return data, split_idx, train_loader_inductive, valid_loader_inductive, test_loader_inductive
