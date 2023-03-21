""" Base Neural Network and Knowledge Enhanced Models """
from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.loader import *
from kegnn.parsers import *
from torch_geometric.loader import GraphSAINTRandomWalkSampler as RWSampler
import Transforms as T
import importlib
import sys, inspect
from preprocess_data import PygDataset
from knowledge import KnowledgeGenerator
from pathlib import Path


def get_model(args):
    """ instantiates the model specified in args """

    msg = f'{args.model} is not implemented. Choose a model in the list: ' \
          f'{[x[0] for x in inspect.getmembers(sys.modules["model"], lambda c: inspect.isclass(c) and c.__module__ == get_model.__module__)]}'
    module = importlib.import_module("model")
    try:
        _class = getattr(module, args.model)
    except AttributeError:
        raise NotImplementedError(msg)

    return _class(args)


class _GraphSampling(torch.nn.Module):
    """
    Super Class for sampling batches in mini-batch training
    Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
    and from here https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking/blob/main/GraphSampling/_GraphSampling.py
    """

    def __init__(self, args):
        super(_GraphSampling, self).__init__()
        self.data = PygDataset(args).data
        self.num_neighbors = args.num_neighbors
        self.train_data = T.DropTrainEdges(args)(PygDataset(args).data)
        self.batch_size = args.batch_size
        self.hidden_channels = args.hidden_channels
        self.num_features = self.data.num_features
        self.out_channels = self.data.num_classes
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.num_workers = args.num_workers
        self.num_layers_sampling = args.num_layers_sampling
        self.full_batch = args.full_batch

        self.test_loader = NeighborLoader(self.data,
                                          num_neighbors=self.num_neighbors[:self.num_layers_sampling],
                                          shuffle=False,  # order needs to be respected here
                                          input_nodes=None,
                                          batch_size=len(self.data.test_mask) if self.full_batch else self.batch_size,
                                          num_workers=self.num_workers,
                                          transform=T.RelationsAttribute(),
                                          neighbor_sampler=None)

    def __new__(cls, *args, **kwargs):
        """ avoid instantiation without subclass """
        if cls is _GraphSampling:
            raise TypeError(f'{cls.__name__} can only be baseclass and must not be instantiated without subclass.')
        return super().__new__(cls)

    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def reset_parameters(self, **kwargs):
        pass


class GAT(_GraphSampling):
    """ Implementation of GAT """

    def __init__(self, args, **kwargs):
        super(GAT, self).__init__(args)
        self.name = 'GAT'
        self.in_head = args.attention_heads
        self.out_head = 1
        self.convs = torch.nn.ModuleList()
        self.conv1 = GATConv(self.num_features, self.hidden_channels, heads=self.in_head, dropout=self.dropout)

        for _ in range(self.num_layers - 2):
            conv = GATConv(self.hidden_channels * self.in_head, self.hidden_channels, heads=self.in_head,
                           dropout=self.dropout)
            self.convs.append(conv)

        self.conv2 = GATConv(self.hidden_channels * self.in_head, self.out_channels, concat=False,
                             heads=self.out_head, dropout=self.dropout)
        self.train_loader = NeighborLoader(self.train_data,
                                           num_neighbors=self.num_neighbors[:self.num_layers_sampling],
                                           shuffle=True,
                                           input_nodes=None,
                                           batch_size=len(self.data.train_mask) if self.full_batch else self.batch_size,
                                           num_workers=self.num_workers,
                                           transform=T.RelationsAttribute(),
                                           neighbor_sampler=None)

    def reset_parameters(self, **kwargs):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight):
        # todo verify with dropout, elu etc.
        # to do also very slow
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCN(_GraphSampling):
    """
    GCN module baseline given by OGB
    """

    def __init__(self, args, **kwargs):
        super(GCN, self).__init__(args)
        self.name = 'GCN'
        self.convs = ModuleList()
        self.convs.append(GCNConv(self.num_features, self.hidden_channels))
        self.bns = ModuleList()
        self.bns.append(BatchNorm1d(self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GCNConv(self.hidden_channels, self.hidden_channels))
            self.bns.append(BatchNorm1d(self.hidden_channels))
        self.convs.append(GCNConv(self.hidden_channels, self.out_channels))
        # self.lin = Linear(self.hidden_channels, self.out_channels)

        self.train_loader = NeighborLoader(self.train_data,
                                           num_neighbors=self.num_neighbors[:self.num_layers_sampling],
                                           shuffle=True,
                                           input_nodes=None,
                                           batch_size=len(self.data.train_mask) if args.full_batch else self.batch_size,
                                           num_workers=self.num_workers,
                                           transform=T.RelationsAttribute(),
                                           neighbor_sampler=None)

    def reset_parameters(self):
        # self.lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x



class MLP(_GraphSampling):
    """ MLP baseline for OGB """

    def __init__(self, args, **kwargs):
        super(MLP, self).__init__(args)
        self.name = 'MLP'

        self.lins = ModuleList()
        self.lins.append(Linear(self.num_features, self.hidden_channels))
        self.bns = ModuleList()
        self.bns.append(BatchNorm1d(self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.lins.append(Linear(self.hidden_channels, self.hidden_channels))
            self.bns.append(BatchNorm1d(self.hidden_channels))
        self.lins.append(Linear(self.hidden_channels, self.out_channels))

        self.train_loader = NeighborLoader(self.train_data,
                                           num_neighbors=self.num_neighbors[:self.num_layers_sampling],
                                           shuffle=True,
                                           input_nodes=None,
                                           batch_size=len(self.data.train_mask) if self.full_batch else self.batch_size,
                                           num_workers=self.num_workers,
                                           transform=T.RelationsAttribute(),
                                           neighbor_sampler=None)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class KENN_GCN(GCN):
    """ kenn-sub with GCN as base NN"""

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()
        self.clause_weight = args.clause_weight

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge, initial_clause_weight=args.clause_weight,
                                                      boost_function=args.boost_function))

    def reset_parameters(self):
        super().reset_parameters()  # should call reset parameter function of MLP
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight)
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)
        return z


class KENN_MLP(MLP):
    """ kenn-sub with MLP (from ogb) as base NN"""

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()
        self.clause_weight = args.clause_weight

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge, initial_clause_weight=args.clause_weight,
                                                      boost_function=args.boost_function))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)

        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)
        return z


class KENN_GAT(GAT):
    """ kenn-sub with GAT as base NN """

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge, initial_clause_weight=args.clause_weight,
                                                      boost_function=args.boost_function))

    def reset_parameters(self):
        super().reset_parameters()  # should call reset parameter function of MLP
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)

        return z



