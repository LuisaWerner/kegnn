import torch
from torch_geometric.loader import NeighborLoader, ClusterLoader, ClusterData

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

    if args.cluster_sampling:
        # todo: not sure yet if this is inductive or transductive setting, appy different way to handle inductive vs transductive
        # asked this question at https://github.com/pyg-team/pytorch_geometric/discussions/5030
        train_loader = ClusterLoader(data=ClusterData(data.subgraph(split_idx['train']), num_parts=args.num_partitions),
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)
        valid_loader = ClusterLoader(data=ClusterData(data.subgraph(split_idx['valid']), num_parts=args.num_partitions),
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

        test_loader = ClusterLoader(data=ClusterData(data.subgraph(split_idx['test']), num_parts=args.num_partitions),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)

        return train_loader, valid_loader, test_loader

    else:
        train_loader = NeighborLoader(data=data.subgraph(split_idx['train']) if args.mode == 'inductive' else data,
                                      num_neighbors=[args.sampling_neighbor_size] * args.num_layers_sampling,
                                      shuffle=True,
                                      input_nodes=None,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers)

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
