import torch
import torch_sparse

from ogb.nodeproppred import PygNodePropPredDataset


def load_and_preprocess(args):
    """
    @param args: Argument Parser object specified by user
    loads dataset from ogb database
    modifies the data object
    generates inductive links
    """
    dataset = PygNodePropPredDataset(name=args.dataset)
    data = dataset[0]
    data.num_classes = dataset.num_classes
    data.adj_t = torch_sparse.SparseTensor(row=data.edge_index[0], col=data.edge_index[1],
                                           sparse_sizes=(data.num_nodes, data.num_nodes))
    data.relations = torch.full(size=(data.num_edges, 1), fill_value=args.binary_preactivation)

    split_idx = dataset.get_idx_split()

    if args.inductive:
        mask_train = torch.all(torch.isin(data.edge_index, split_idx['train']), dim=0)
        mask_valid = torch.all(torch.isin(data.edge_index, split_idx['valid']), dim=0)
        mask_test = torch.all(torch.isin(data.edge_index, split_idx['test']), dim=0)

        data.adj_train = torch_sparse.SparseTensor(row=torch.masked_select(input=data.edge_index[0], mask=mask_train),
                                                   col=torch.masked_select(input=data.edge_index[1], mask=mask_train),
                                                   sparse_sizes=(data.num_nodes, data.num_nodes))

        data.adj_valid = torch_sparse.SparseTensor(row=torch.masked_select(input=data.edge_index[0], mask=mask_valid),
                                                   col=torch.masked_select(input=data.edge_index[1], mask=mask_valid),
                                                   sparse_sizes=(data.num_nodes, data.num_nodes))

        data.adj_test = torch_sparse.SparseTensor(row=torch.masked_select(input=data.edge_index[0], mask=mask_test),
                                                  col=torch.masked_select(input=data.edge_index[1], mask=mask_test),
                                                  sparse_sizes=(data.num_nodes, data.num_nodes))

        data.relations_train = torch.full(size=(torch.count_nonzero(mask_train).item(), 1),
                                          fill_value=args.binary_preactivation)
        data.relations_valid = torch.full(size=(torch.count_nonzero(mask_valid).item(), 1),
                                          fill_value=args.binary_preactivation)
        data.relations_test = torch.full(size=(torch.count_nonzero(mask_test).item(), 1),
                                         fill_value=args.binary_preactivation)

    return data, split_idx
