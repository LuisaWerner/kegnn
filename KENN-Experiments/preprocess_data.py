import torch
from torch_geometric.loader import *
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset


def create_batches(data, split_idx, args):
    """
    @param data: Input graph PygNodePropPredDataset
    @param split_idx: indices for train, valid test
    @param args: hyper parameters
    Creates batches for training, validation and test given the procedure defined in the arguments.
    If not specified, follow GraphSAGE sampling
    """

    if args.batch_size > data.num_nodes or args.full_batch:
        print('Full batch training ')
        args.batch_size = data.num_nodes

    if args.graph_saint:
        train_loader = GraphSAINTNodeSampler(data, batch_size=args.batch_size)

    elif args.cluster_sampling:
        # todo: not sure yet if this is inductive or transductive setting, appy different way to handle inductive vs transductive
        # asked this question at https://github.com/pyg-team/pytorch_geometric/discussions/5030
        train_loader = ClusterLoader(
            data=ClusterData(data.subgraph(split_idx['train']) if args.mode == 'inductive' else data,
                             num_parts=args.cluster_sampling_num_partitions),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers)

    else:
        train_loader = NeighborLoader(data=data.subgraph(split_idx['train']) if args.mode == 'inductive' else data,
                                      num_neighbors=[args.sampling_neighbor_size] * args.num_layers_sampling,
                                      shuffle=True,
                                      input_nodes=None,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers)

    # always take Neighbor loader for valid and test
    valid_loader = NeighborLoader(data=data.subgraph(split_idx['valid']) if args.mode == 'inductive' else data,
                                  num_neighbors=[args.sampling_neighbor_size] * args.num_layers_sampling,
                                  shuffle=True,
                                  input_nodes=None,
                                  batch_size=args.batch_size)

    test_loader = NeighborLoader(data=data.subgraph(split_idx['test']) if args.mode == 'inductive' else data,
                                 num_neighbors=[args.sampling_neighbor_size] * args.num_layers_sampling,
                                 shuffle=True,
                                 input_nodes=None,
                                 batch_size=args.batch_size)

    return train_loader, valid_loader, test_loader


def get_mask(split_idx, data):
    """
    todo
    """
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask
    return data


def to_inductive(data):
    """
    todo: find another solution for inductive/transductive that is more intuitive
    """
    data = data.clone()
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
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
    split_idx = dataset.get_idx_split()

    train_loader, valid_loader, test_loader = create_batches(data, split_idx, args)

    return data, split_idx, train_loader, valid_loader, test_loader
