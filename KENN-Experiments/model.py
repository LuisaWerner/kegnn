""" Base Neural Network and Knowledge Enhanced Models """
from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.loader import *
from kenn.parsers import *
from torch_geometric.loader import GraphSAINTRandomWalkSampler as RWSampler
import Transforms as T
import importlib
import sys, inspect
from preprocess_data import PygDataset
from knowledge import KnowledgeGenerator


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
        self.inductive = True if args.mode == 'inductive' else False
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


class LinearRegression(_GraphSampling):
    def __init__(self, args, **kwargs):
        super(LinearRegression, self).__init__(args)
        self.name = 'LinearRegression'
        self.lin = Linear(self.num_features, self.out_channels)
        self.train_loader = NeighborLoader(T.ToInductive()(self.train_data) if self.inductive else self.train_data,
                                           num_neighbors=self.num_neighbors[:self.num_layers_sampling],
                                           shuffle=True,
                                           input_nodes=None,
                                           batch_size=len(self.data.train_mask) if self.full_batch else self.batch_size,
                                           num_workers=self.num_workers,
                                           transform=T.RelationsAttribute(),
                                           neighbor_sampler=None)

    def reset_parameters(self, **kwargs):
        self.lin.reset_parameters()

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        x = self.lin(x)
        return x


class LogisticRegression(_GraphSampling):
    def __init__(self, args, **kwargs):
        super(LogisticRegression, self).__init__(args)
        self.name = 'LinearRegression'
        self.lin = Linear(self.num_features, self.out_channels)
        self.train_loader = NeighborLoader(T.ToInductive()(self.train_data) if self.inductive else self.train_data,
                                           num_neighbors=self.num_neighbors[:self.num_layers_sampling],
                                           shuffle=True,
                                           input_nodes=None,
                                           batch_size=len(self.data.train_mask) if self.full_batch else self.batch_size,
                                           num_workers=self.num_workers,
                                           transform=T.RelationsAttribute(),
                                           neighbor_sampler=None)

    def reset_parameters(self, **kwargs):
        self.lin.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight):
        x = torch.sigmoid(self.lin(x))
        return x


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
        self.train_loader = NeighborLoader(T.ToInductive()(self.train_data) if self.inductive else self.train_data,
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


class ClusterGCN(_GraphSampling):
    def __init__(self, args, **kwargs):
        super(ClusterGCN, self).__init__(args)
        self.name = 'ClusterGCN'
        self.parts = args.num_parts
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(self.num_features, self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(self.hidden_channels, self.hidden_channels))
        self.convs.append(SAGEConv(self.hidden_channels, self.out_channels))

        sample_size = max(1, int(self.batch_size / (self.train_data.num_nodes / self.num_parts)))
        cluster_data = ClusterData(T.ToInductive()(self.train_data) if self.inductive else self.train_data,
                                   num_parts=self.num_parts, recursive=False)
        self.train_loader = ClusterLoader(cluster_data, batch_size=sample_size, shuffle=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        for i, conv in enumerate(self.convs):

            x = conv(x, edge_index)  # no edge weight for SAGEConv
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class KENN_ClusterGCN(ClusterGCN):
    """ kenn-sub with GraphSage (from ogb) as base NN"""

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge))

    def reset_parameters(self):
        """ resets parameters to default initialization: Base NN and KENN clause weights """
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)

        return z


class GraphSAINT(_GraphSampling):
    # Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
    def __init__(self, args, **kwargs):
        super(GraphSAINT, self).__init__(args)
        self.name = 'GraphSAINT'
        self.num_steps = args.num_steps
        self.walk_length = args.walk_length
        self.use_norm = args.use_norm
        self.sample_covreage = args.sample_coverage
        self.aggr = "add" if self.use_norm else "mean"
        self.convs = ModuleList()
        self.convs.append(SAGEConv(self.num_features, self.hidden_channels))
        for _ in range(self.num_layers - 2):
            conv = SAGEConv(self.hidden_channels, self.hidden_channels)
            conv.aggr = self.aggr
            self.convs.append(conv)
        self.lin = Linear(self.hidden_channels, self.out_channels)

        self.train_loader = RWSampler(
            T.ToInductive()(self.train_data) if self.inductive else self.train_data,
            batch_size=self.batch_size,
            walk_length=self.walk_length,
            num_steps=self.num_steps,
            num_workers=self.num_workers,
            sample_coverage=self.sample_coverage,
        )

    def reset_parameters(self):
        self.lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)  # no edge_weight for SAGEConv
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)  # todo is self.training activated?
        x = self.lin(x)
        return x


class KENN_SAINT(GraphSAINT):
    """ kenn-sub with GraphSage (from ogb) as base NN"""

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge))

    def reset_parameters(self):
        """ resets parameters to default initialization: Base NN and KENN clause weights """
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)

        return z


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

        self.train_loader = NeighborLoader(T.ToInductive()(self.train_data) if self.inductive else self.train_data,
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


class SAGE(_GraphSampling):
    # TODO compare differences to :https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking/blob/6a3b91c7bdd0459f454c92a364bca2a69e26cda4/GraphSampling/GraphSAGE.py
    """ Implementation of GraphSAGE - ogb baseline  """

    def __init__(self, args, **kwargs):
        super(SAGE, self).__init__(args)
        self.name = 'SAGE'
        self.convs = ModuleList()
        self.convs.append(SAGEConv(self.num_features, self.hidden_channels))
        self.bns = ModuleList()
        self.bns.append(BatchNorm1d(self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(self.hidden_channels, self.hidden_channels))
            self.bns.append(BatchNorm1d(self.hidden_channels))
        self.convs.append(SAGEConv(self.hidden_channels, self.out_channels))
        self.train_loader = NeighborLoader(T.ToInductive()(self.train_data),  # always inductive with graphSAGE
                                           num_neighbors=self.num_neighbors[:self.num_layers_sampling],
                                           shuffle=True,
                                           input_nodes=self.train_data.train_mask,  # todo is this needed ?
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           transform=T.RelationsAttribute(),
                                           neighbor_sampler=None)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)  # no edge_weight for SAGEConv
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
        self.train_loader = NeighborLoader(T.ToInductive()(self.train_data) if self.inductive else self.train_data,
                                           num_neighbors=self.num_neighbors[:self.num_layers_sampling],
                                           shuffle=True,
                                           input_nodes=None,
                                           batch_size=self.batch_size,
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


class Standard(_GraphSampling):
    """ Original Base Neural Network by kenn-sub paper """

    hidden_channels = 50
    out_channels = 6
    dropout = 0.5

    def __init__(self, args, **kwargs):
        super(Standard, self).__init__(args)
        self.name = 'Standard'
        self.lin_layers = ModuleList()
        self.lin_layers.append(Linear(self.in_channels, self.hidden_channels))
        self.lin_layers.append(Linear(self.hidden_channels, self.hidden_channels))
        self.lin_layers.append(Linear(self.hidden_channels, self.hidden_channels))
        self.lin_layers.append(Linear(self.hidden_channels, self.out_channels))
        self.train_loader = NeighborLoader(T.ToInductive()(self.train_data) if self.inductive else self.train_data,
                                           num_neighbors=self.num_neighbors[:self.num_layers_sampling],
                                           shuffle=True,
                                           input_nodes=None,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           transform=T.RelationsAttribute(),
                                           neighbor_sampler=None)

    def reset_parameters(self):
        for layer in self.lin_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index=None, relations=None, edge_weight=None):
        for i, lin in enumerate(self.lin_layers[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_layers[-1](x)
        return x


class KENN_GCN(GCN):
    """ kenn-sub with GCN as base NN"""

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        # self.knowledge_file = knowledge_file
        self.knowledge = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge))

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

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)
        return z


class KENN_SAGE(SAGE):
    """ kenn-sub with GraphSage (from ogb) as base NN"""

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge))

    def reset_parameters(self):
        super().reset_parameters()  # should call reset parameter function of MLP
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)
        return z


class KENN_GAT(GAT):
    """ kenn-sub with GraphSage (from ogb) as base NN"""

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge))

    def reset_parameters(self):
        super().reset_parameters()  # should call reset parameter function of MLP
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)

        return z


class KENN_LogisticRegression(LogisticRegression):
    """ kenn-sub with MLP (from ogb) as base NN"""

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)
        return z


class KENN_LinearRegression(LinearRegression):
    """ kenn-sub with MLP (from ogb) as base NN"""

    def __init__(self, args):
        super().__init__(args)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = KnowledgeGenerator(self, args).knowledge
        self.kenn_layers = ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=self.knowledge))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=edge_weight)
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)
        return z
