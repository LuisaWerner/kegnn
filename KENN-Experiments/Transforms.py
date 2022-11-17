from torch_geometric.transforms import BaseTransform, ToUndirected
from torch_geometric.utils import *


class RelationsAttribute(BaseTransform):
    """ makes sure that the tensor with binary preactivations for kenn-sub binary predicates is of correct size """

    def __call__(self, data):
        num_edges = data.edge_index.shape[1]
        data.relations = data.relations[:num_edges]
        return data


class ToInductive(BaseTransform):
    """
    Prepares data object for inductive training
    Full dataset stays the same, simply the nodes in edge_index not in train are deleted.
    """

    def __call__(self, data):
        mask = data.train_mask
        data.edge_index, _ = subgraph(mask, data.edge_index, None, relabel_nodes=False, num_nodes=data.num_nodes)
        data.relations = data.relations[:data.edge_index.shape[1]]
        return data