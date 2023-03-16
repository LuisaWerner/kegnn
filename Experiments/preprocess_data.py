import torch
import torch_geometric.datasets
import Transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np


def compute_compliance(model):
    """
    compute clause compliance per iteration
    returns a list of length |number classes| with compliance value per each
    """
    # y = model.data.y.cpu().detach().numpy() do we need to detach or is this even slowing down the code?
    y = model.data.y.numpy()
    edge_index = np.transpose(model.data.edge_index.numpy())
    train_mask = model.data.train_mask.numpy()
    train_edge_mask = np.logical_or(train_mask[edge_index[:, 0]], train_mask[edge_index[:, 1]])

    # calculate the classes corresponding to edge index
    edge_index_cls = np.zeros_like(edge_index)
    for row in range(edge_index.shape[0]):
        edge_index_cls[row, 0] = y[edge_index[row, 0]]
        edge_index_cls[row, 1] = y[edge_index[row, 1]]

    compliance = []
    for cls in range(model.data.num_classes):
        cls_mask = np.logical_or(edge_index_cls[:, 0] == cls, edge_index_cls[:, 1] == cls)
        mask = np.logical_and(cls_mask, train_edge_mask) # edges that have at least a training node and a node of class cls
        same_mask = np.logical_and(mask, np.equal(edge_index_cls[:, 0], edge_index_cls[:, 1])) # edges that are of the above set and have the same class for both nodes
        cls_compliance = sum(same_mask)/sum(mask)
        compliance.append(cls_compliance)

    return compliance

class PygDataset:
    """ loads the dataset depending on the name """

    planet_sets = ['CiteSeer', 'Cora', 'PubMed']
    ogbn = ['ogbn-products', 'ogbn-arxiv']
    saint_datasets = ["Reddit2", "Flickr", "AmazonProducts", "Yelp"]

    def __init__(self, args):

        if args.undirected:
            transform = T.Compose([T.ToUndirected(), T.AddAttributes(args)])
        else:
            transform = T.AddAttributes(args)

        if args.dataset in self.planet_sets:
            _dataset = torch_geometric.datasets.Planetoid(root=args.dataset, name=args.dataset, split=args.planetoid_split, transform=transform)
        elif args.dataset in self.ogbn:
            _dataset = PygNodePropPredDataset(name=args.dataset, transform=transform)
        elif args.dataset in self.saint_datasets:
            _dataset = getattr(torch_geometric.datasets, args.dataset)(root=args.dataset, transform=transform)
        else:
            raise ValueError(f'Unknown dataset {args.dataset} specified. Use one out of: {self.planet_sets + self.ogbn + self.saint_datasets}')

        [self._data] = _dataset

        if not hasattr(self._data, "train_mask"):
            split_dict = _dataset.get_idx_split()
            split_dict['val'] = split_dict.pop('valid')
            for key, idx in split_dict.items():
                mask = torch.zeros(self._data.num_nodes, dtype=torch.bool)
                mask[idx] = True
                self._data[f'{key}_mask'] = mask

        self._data.name = args.dataset

    @property
    def data(self):
        return self._data









