
import torch
import torch_geometric.datasets
import Transforms as T
from ogb.nodeproppred import PygNodePropPredDataset


class PygDataset:
    """ loads the dataset depending on the name """

    planet_sets = ['CiteSeer', 'Cora', 'PubMed']
    ogbn = ['ogbn-products', 'ogbn-arxiv']
    saint_datasets = ["Reddit2", "Flickr", "AmazonProducts", "Yelp"]

    def __init__(self, args):
        if args.dataset in self.planet_sets:
            _dataset = torch_geometric.datasets.Planetoid(root=args.dataset, name=args.dataset, split=args.planetoid_split, transform=T.Compose([T.ToUndirected(), T.AddAttributes(args)]))
        elif args.dataset in self.ogbn:
            _dataset = PygNodePropPredDataset(name=args.dataset, transform=T.Compose([T.ToUndirected(), T.AddAttributes(args)]))
        elif args.dataset in self.saint_datasets:
            _dataset = getattr(torch_geometric.datasets, args.dataset)(root=args.dataset, transform=T.Compose([T.ToUndirected(), T.AddAttributes(args)]))
        else:
            raise ValueError(f'Unknown dataset specified. Use one out of: {self.planet_sets + self.ogbn + self.saint_datasets}')

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









