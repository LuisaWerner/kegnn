import torch
from torch_geometric.loader import *
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph

from ogb.nodeproppred import PygNodePropPredDataset


class RelationsAttribute(BaseTransform):
    """ makes sure that the tensor with binary preactivations for KENN binary predicates is of correct size """

    def __call__(self, data):
        num_edges = data.edge_index.shape[1]
        data.relations = data.relations[:num_edges]
        return data


def to_inductive(data):
    """ Prepares data object for inductive training """
    data = data.clone()
    mask = data.train_mask
    data.edge_index, _ = subgraph(mask, data.edge_index, None, relabel_nodes=True, num_nodes=data.num_nodes)
    data.relations = data.relations[:data.edge_index.shape[1]]  # todo this should be done later
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask, data.valid_mask = None, None
    data.node_year = data.node_year[mask]
    data.num_nodes = mask.sum().item()

    return data


def sample_batches(data, args):
    """
    samples batches for testing
    """
    loader = NeighborLoader(data,
                            num_neighbors=[args.sampling_neighbor_size] * args.num_layers_sampling,
                            # todo : depends also on kenn layers/ base NN structure, --> to verify !
                            shuffle=False,  # order needs to be respected here
                            input_nodes=None,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            transform=RelationsAttribute())
    return loader


def sample_train_batches(data, args):
    if args.batch_size > data.num_nodes or args.full_batch:
        print('Full batch training ')
        args.batch_size = data.num_nodes

    if args.mode == 'inductive':
        data = to_inductive(data)

    if args.train_sampling == 'cluster':
        # TODO
        train_loader = ClusterLoader(
            data=ClusterData(data, num_parts=args.cluster_sampling_num_partitions),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            transform=RelationsAttribute())  # todo  doesnt have transform attribute

    elif args.train_sampling == 'graph_saint':
        train_loader = GraphSAINTRandomWalkSampler(data,
                                                   batch_size=args.batch_size,
                                                   walk_length=args.walk_length,
                                                   num_steps=args.num_steps,
                                                   sample_coverage=args.sample_coverage,
                                                   num_workers=args.num_workers
                                                   )

    else:
        "If nothing specified, create batches in the same way as for testing"
        train_loader = NeighborLoader(data,
                                      num_neighbors=[args.sampling_neighbor_size] * args.num_layers_sampling,
                                      # todo : depends also on kenn layers/ base NN structure, --> to verify !
                                      shuffle=True,
                                      input_nodes=None,  # data.train_mask,  # the target nodes are only from training
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      transform=RelationsAttribute())

    return train_loader


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
    for key, idx in dataset.get_idx_split().items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    train_loader = sample_train_batches(data, args)
    all_loader = sample_batches(data, args)

    return data, train_loader, all_loader
