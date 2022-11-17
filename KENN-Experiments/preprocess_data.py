import os.path
import warnings
import torch
import torch_geometric.datasets
from torch_geometric.loader import *
import Transforms as T
from data_stats import *
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import degree


def get_partition_sizes(cluster_data: ClusterData) -> list:
    """ returns list of sizes of partitions in cluster data """
    part_sizes = []
    for i in range(len(cluster_data.partptr) - 1):
        size = cluster_data.partptr[i + 1] - cluster_data.partptr[i]
        part_sizes.append(size.item())

    return part_sizes


def sample_batches(data: Data, args) -> NeighborLoader:
    """
    samples batches for testing
    """
    loader = NeighborLoader(data,
                            num_neighbors=[args.sampling_neighbor_size] * args.num_layers_sampling,
                            # todo : depends also on kenn-sub layers/ base NN structure, --> to verify !
                            shuffle=False,  # order needs to be respected here
                            input_nodes=None,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            transform=T.RelationsAttribute())
    return loader


def sample_train_batches(data: Data, args) -> DataLoader:
    if args.batch_size > data.num_nodes or args.full_batch:
        print('Full batch training ')
        args.batch_size = data.num_nodes

    if args.train_sampling == 'cluster':
        # TODO SIGSEV error if too many num_parts 100 doesnt pass, 50 passes

        # if partition size is larger than batch size, set partition size to batch size
        if args.cluster_partition_size > args.batch_size:
            warnings.warn('batch size smaller than partition size: use one partition as a batch now')
            args.cluster_partition_size = args.batch_size

        num_partitions = round(data.num_nodes / args.cluster_partition_size) + 1
        cluster_data = ClusterData(data, num_parts=num_partitions, recursive=False)

        partition_sizes = get_partition_sizes(cluster_data)
        avg_partition_size = sum(partition_sizes) / len(partition_sizes)
        print(f'Avg Partition size: {avg_partition_size}')

        train_loader = ClusterLoader(
            cluster_data=cluster_data,
            batch_size=round(args.batch_size / args.cluster_partition_size) + 1,
            shuffle=True,
            num_workers=args.num_workers)

        print(f'# Partitions to form a batch: {round(args.batch_size / args.cluster_partition_size) + 1}')

    elif args.train_sampling == 'graph_saint':
        train_loader = GraphSAINTRandomWalkSampler(data,
                                                   batch_size=args.batch_size,
                                                   walk_length=args.walk_length,
                                                   num_steps=round(data.num_nodes / args.batch_size),
                                                   sample_coverage=args.sample_coverage,
                                                   num_workers=args.num_workers
                                                   )

    else:
        "If nothing specified, create batches in the same way as for testing"
        train_loader = NeighborLoader(data,
                                      num_neighbors=[args.sampling_neighbor_size] * args.num_layers_sampling,
                                      # todo : depends also on kenn-sub layers/ base NN structure, --> to verify !
                                      shuffle=True,
                                      input_nodes=None,  # data.train_mask,  # the target nodes are only from training
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      transform=T.RelationsAttribute())

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
    if args.dataset in ['CiteSeer', 'Cora', 'PubMed']:
        dataset = torch_geometric.datasets.Planetoid(root=args.dataset, name=args.dataset, split=args.planetoid_split)

    elif args.dataset in ['ogbn-products', 'ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name=args.dataset)

    elif args.dataset in ["Reddit2", "Flickr", "AmazonProducts", "Yelp"]:
        dataset = getattr(torch_geometric.datasets, args.dataset)(root=args.dataset)
    else:
        raise ValueError('Unknown dataset specified. Use one out of: '
                         '{CiteSeer, Cora, Pubmed , ogbn-products, ogbn-arxiv, Reddit, Flickr, AmazonProducts, Yelp}')

    data = dataset[0]
    data = T.ToUndirected()(data)  # this is needed for
    data.num_classes = dataset.num_classes
    data.relations = torch.full(size=(data.num_edges, 1), fill_value=args.binary_preactivation)

    if dataset.data.y.dim() == 1:
        data.y = data.y.unsqueeze(1)

    # Convert split indices to boolean masks and add them to `data`.
    if not hasattr(data, "train_mask"):
        split_dict = dataset.get_idx_split()
        split_dict['val'] = split_dict.pop('valid')
        for key, idx in split_dict.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f'{key}_mask'] = mask

    if not hasattr(data, "num_nodes"):
        data.num_nodes = data.x.shape[0]

    if not hasattr(data, "num_features"):
        data.num_featuers = data.x.shape[1]

    # create edge weight with ones if there's no edge weight stored by default
    if data.edge_weight is None:
        # data.edge_weight = torch.ones(data.edge_index.size()[1])
        row, col = data.edge_index
        data.edge_weight = 1.0 / degree(col, data.num_nodes)[col]

    # train_loader = sample_train_batches(data, args)
    # all_loader = sample_batches(data, args)

    # if args.save_data_stats:
    if args.save_data_stats and not os.path.exists('data_stats'):
        print('Saving Data Stats..... ')
        save_data_stats(data, args) # todo does this work for Reddit etc.?

    data.n_id = torch.arange(data.num_nodes) # store original node ids # todo needed?

    return data  # , train_loader, all_loader # todo do the loaders in the model class
