from torch_geometric.transforms import BaseTransform, ToUndirected, Compose
from torch_geometric.utils import *
import torch
import random


class RelationsAttribute(BaseTransform):
    """ makes sure that the tensor with binary preactivations for kenn-sub binary predicates is of correct size """

    def __call__(self, data):
        data.relations = data.relations[:data.num_edges]
        return data


class DropTrainEdges(BaseTransform):
    """ for training: randomly drops edges in training mode """

    def __init__(self, args):
        super(DropTrainEdges, self).__init__()
        self.edges_drop_rate = args.edges_drop_rate

    def __setattr__(self, key, value):
        if value >= 1.0:
            raise AttributeError(f'{key} has to be smaller than one. Otherwise all edges are deleted')
        else:
            super().__setattr__(key, value)

    def __call__(self, data):
        edge_mask = random.choices([True, False], weights=[1-self.edges_drop_rate, self.edges_drop_rate], k=data.num_edges)
        data.edge_index = data.edge_index[:, edge_mask]
        data.edge_weight = data.edge_weight[edge_mask]
        data.relations = data.relations[edge_mask]
        return data


class AddAttributes(BaseTransform):
    """ Adds missing attributes to data object"""

    def __init__(self, args):
        self.args = args

    def __call__(self, data, *args, **kwargs):
        if data.y.dim() == 1:
            data.y = data.y.unsqueeze(1)

        data.num_classes = len(torch.unique(data.y))

        data.relations = torch.full(size=(data.num_edges, 1), fill_value=self.args.binary_preactivation)

        if not hasattr(data, "num_nodes"):
            data.num_nodes = data.x.shape[0]

        if not hasattr(data, "num_features"):
            data.num_features = data.x.shape[1]

        if not hasattr(data, "num_edges"):
            data.num_edges = data.edge_index[1]

        if data.edge_weight is None:
            data.edge_weight = torch.ones(data.edge_index.size()[1])
        else:
            UserWarning('Dataset has weighted edges that might require treatment such as normalization')

        if self.args.normalize_edges:
            row, col = data.edge_index
            data.edge_weight = data.edge_weight[col] / degree(col, data.num_nodes)[col]

        data.n_id = torch.arange(data.num_nodes)

        return data
