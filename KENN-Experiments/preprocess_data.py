import torch
from torch_geometric.loader import *
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset


def sample_batches(data, args):
    """
    samples batches for testing
    """
    loader = NeighborLoader(data,
                            num_neighbors=[args.sampling_neighbor_size] * args.num_layers_sampling,
                            shuffle=True,
                            input_nodes=None,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
    return loader


def sample_train_batches(data, args):
    if args.batch_size > data.num_nodes or args.full_batch:
        print('Full batch training ')
        args.batch_size = data.num_nodes

    if args.mode == 'inductive':
        data = to_inductive(data)

    if args.train_sampling == 'cluster':
        train_loader = ClusterLoader(
            data=ClusterData(data, num_parts=args.cluster_sampling_num_partitions),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)

    elif args.train_sampling == 'graph_saint':
        train_loader = GraphSAINTRandomWalkSampler(data,
                                                   batch_size=args.batch_size,
                                                   walk_length=args.walk_length,
                                                   num_steps=args.num_steps,
                                                   sample_coverage=args.sample_coverage,
                                                   # todo : if sample coverage set to 0, loader contains no
                                                   # normalization coefficients
                                                   num_workers=args.num_workers)

    else:
        "If nothing specified, create batches in the same way as for testing"
        train_loader = sample_batches(data, args)

    return train_loader


def to_inductive(data):
    """
    makes training set inductive by removing links to valid and test nodes
    """
    data = data.clone()
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None, relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


def load_and_preprocess(args):
    """
    loads data and create batches
    In the inductive setting, the data object is reduced before to avoid links between
    different subsets. Only neighbors from the respective subset are sampled
    @param args: Argument Parser object specified by user
    @return train_loader, valid_loader, test_loader: train/valid/test set split into batches,
    samples neighbors specified in args in an transductive
    """
    dataset = PygNodePropPredDataset(name=args.dataset)
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.relations = torch.full(size=(data.num_edges, 1), fill_value=args.binary_preactivation)

    # Convert split indices to boolean masks and add them to `data`.
    split_idx = dataset.get_idx_split()
    for key, idx in dataset.get_idx_split().items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    train_loader = sample_train_batches(data, args)
    test_loader = sample_batches(data, args)

    return data, split_idx, train_loader, test_loader
