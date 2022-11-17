""" Base Neural Network and Knowledge Enhanced Models """
from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, Linear
from torch_geometric.loader import *
from kenn.parsers import *
from torch_geometric.loader import GraphSAINTRandomWalkSampler as RWSampler
import Transforms as T


def get_model(data, args):
    """ instantiates the model specified in args """

    msg = f'{args.model} is not implemented. Choose a model in the list: GCN, SAGE, MLP, SAINT, KENN_GCN, KENN_SAGE, KENN_MLP, KENN_SAINT'

    # Base neural network
    if not args.model.startswith('KENN'):
        if args.model == 'MLP':
            _class = MLP
        elif args.model == 'GCN':
            _class = GCN
        elif args.model == 'SAGE':
            _class = SAGE
        elif args.model == 'SAINT':
            _class = GraphSAINT
        elif args.model == 'CLUSTER':
            _class = ClusterGCN
        else:
            NotImplementedError(msg)
            return None

        model = _class(data, args)

    # kenn-sub network
    elif args.model.startswith('KENN'):
        if args.model == 'KENN_MLP':
            _class = KENN_MLP
        elif args.model == 'KENN_GCN':
            _class = KENN_GCN
        elif args.model == 'KENN_SAGE':
            _class = KENN_SAGE
        elif args.model == 'KENN_SAINT':
            _class = KENN_SAINT
        elif args.model == 'KENN_CLUSTER':
            _class = KENN_ClusterGCN
        else:
            NotImplementedError(msg)
            return None

        model = _class(data, args, knowledge_file='knowledge_base')
    else:
        NotImplementedError(msg)
        model = None

    return model


class _GraphSampling(torch.nn.Module):
    """
    Super Class for sampling batches in mini-batch training
    Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
    and from here https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking/blob/main/GraphSampling/_GraphSampling.py
    """

    def __init__(self, data, args):
        super(_GraphSampling, self).__init__()
        # define parameter and structures required for all nodes
        self.batch_size = args.batch_size
        self.inductive = True if args.mode == 'inductive' else False
        self.hidden_channels = args.hidden_channels
        self.num_features = data.num_features
        self.out_channels = data.num_classes
        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.num_workers = args.num_workers
        self.sampling_neighbor_size = args.sampling_neighbor_size
        self.num_layers_sampling = args.num_layers_sampling
        self.test_loader = NeighborLoader(data,
                                          num_neighbors=[self.sampling_neighbor_size] * self.num_layers_sampling, # needed?
                                          shuffle=False,  # order needs to be respected here
                                          input_nodes=None,
                                          batch_size=self.batch_size,
                                          num_workers=self.num_workers,
                                          transform=T.RelationsAttribute(),
                                          neighbor_sampler=None)

    @abstractmethod
    def forward(self, **kwargs):
        pass

    @abstractmethod
    def reset_parameters(self, **kwargs):
        pass


class ClusterGCN(_GraphSampling):
    def __init__(self, data, args, **kwargs):
        super(ClusterGCN, self).__init__(data, args)
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(self.num_features, self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(self.hidden_channelsn, self.hidden_channels))
        self.convs.append(SAGEConv(self.hidden_channelsn, self.out_channels))

        sample_size = max(1, int(self.batch_size / (data.num_nodes / args.num_parts)))
        cluster_data = ClusterData(T.ToInductive()(data) if self.to_inductive else data,
                                   num_parts=args.num_parts, recursive=False, save_dir=self.save_dir)
        self.train_loader = ClusterLoader(cluster_data, batch_size=sample_size, shuffle=True)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class KENN_ClusterGCN(ClusterGCN):
    """ kenn-sub with GraphSage (from ogb) as base NN"""
    def __init__(self, data, args, knowledge_file):
        super().__init__(data, args)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = knowledge_file
        self.kenn_layers = torch.nn.ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=knowledge_file))

    def reset_parameters(self):
        """ resets parameters to default initialization: Base NN and KENN clause weights """
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=None)
        # call kenn-sub layers
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)

        return z.log_softmax(dim=-1)


class GraphSAINT(_GraphSampling):
    # Implemented base on https://github.com/rusty1s/pytorch_geometric/blob/master/examples/graph_saint.py
    def __init__(self, data, args, **kwargs):
        super(GraphSAINT, self).__init__(data, args)
        # only define GraphSAINT specific structures here
        self.name = 'GraphSAINT'
        self.use_norm = args.use_norm
        self.aggr = "add" if self.use_norm else "mean"
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(self.num_features, self.hidden_channels))
        for _ in range(self.num_layers - 2):
            conv = SAGEConv(self.hidden_channels, self.hidden_channels)
            conv.aggr = self.aggr
            self.convs.append(conv)
        self.lin = Linear(self.hidden_channels, self.out_channels)

        self.train_loader = RWSampler(
            T.ToInductive()(data) if self.inductive else data,
            batch_size=self.batch_size,
            walk_length=args.walk_length,
            num_steps=args.num_steps,
            num_workers=self.num_workers,
            sample_coverage=args.sample_coverage,
        )

    def reset_parameters(self):
        self.lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training) # todo is self.training activated?
        x = self.lin(x)
        return x


class KENN_SAINT(GraphSAINT):
    """ kenn-sub with GraphSage (from ogb) as base NN"""
    def __init__(self, data, args, knowledge_file):
        super().__init__(data, args)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = knowledge_file
        self.kenn_layers = torch.nn.ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=knowledge_file))

    def reset_parameters(self):
        """ resets parameters to default initialization: Base NN and KENN clause weights """
        super().reset_parameters()  # should call reset parameter function of MLP
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight=None)
        # call kenn-sub layers
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)

        return z.log_softmax(dim=-1)


class GCN(_GraphSampling):
    """
    GCN module baseline given by OGB
    """
    def __init__(self, data, args, **kwargs):
        super(GCN, self).__init__(data, args)
        self.name = 'GCN'
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(self.num_features, self.hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GCNConv(self.hidden_channels, self.hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(self.hidden_channels))
        self.convs.append(GCNConv(self.hidden_channels, self.out_channels))
        # self.lin = Linear(self.hidden_channels, self.out_channels)
        self.train_loader = NeighborLoader(T.ToInductive()(data) if self.inductive else data,
                                          num_neighbors=[self.sampling_neighbor_size] * self.num_layers_sampling,
                                          shuffle=True,
                                          input_nodes=None,
                                          batch_size=self.batch_size,
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
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)


class SAGE(_GraphSampling):
        # TODO compare differences to :https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking/blob/6a3b91c7bdd0459f454c92a364bca2a69e26cda4/GraphSampling/GraphSAGE.py
    """ Implementation of GraphSAGE - ogb baseline  """
    def __init__(self, data, args, **kwargs):
        super(SAGE, self).__init__(data, args)
        self.name = 'SAGE'
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(self.num_features, self.hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.convs.append(SAGEConv(self.hidden_channels, self.hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(self.hidden_channels))
        self.convs.append(SAGEConv(self.hidden_channels, self.out_channels))
        num_neighbors = [25, 10, 5, 5, 5, 5, 5, 5, 5] # todo put in arguments
        self.train_loader = NeighborLoader(T.ToInductive()(data), # always inductive with graphSAGE
                                           num_neighbors=num_neighbors[:self.num_layers_sampling],
                                           shuffle=True,
                                           input_nodes=None,
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
            x = conv(x, edge_index, edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)


class MLP(_GraphSampling):
    """ MLP baseline for OGB """
    def __init__(self, data, args, **kwargs):
        super(MLP, self).__init__(data, args)
        self.name = 'MLP'
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(self.num_features, self.hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(self.hidden_channels))
        for _ in range(self.num_layers - 2):
            self.lins.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(self.hidden_channels))
        self.lins.append(torch.nn.Linear(self.hidden_channels, self.out_channels))
        self.train_loader = NeighborLoader(T.ToInductive()(data) if self.inductive else data,
                                           num_neighbors=[self.sampling_neighbor_size] * self.num_layers_sampling,
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
        return x.log_softmax(dim=-1)


class Standard(_GraphSampling):
    """ Original Base Neural Network by kenn-sub paper """

    hidden_channels = 50
    out_channels = 6
    dropout = 0.5

    def __init__(self, data, args, **kwargs):
        super(Standard, self).__init__(data, args)
        self.name = 'Standard'
        self.lin_layers = torch.nn.ModuleList()
        self.lin_layers.append(torch.nn.Linear(self.in_channels, self.hidden_channels))
        self.lin_layers.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lin_layers.append(torch.nn.Linear(self.hidden_channels, self.hidden_channels))
        self.lin_layers.append(torch.nn.Linear(self.hidden_channels, self.out_channels))
        self.train_loader = NeighborLoader(T.ToInductive()(data) if self.inductive else data,
                                           num_neighbors=[self.sampling_neighbor_size] * self.num_layers_sampling,
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
        # return F.softmax(x, dim=-1)
        return x.log_softmax(dim=-1)


class KENN_GCN(GCN):
    """ kenn-sub with GCN as base NN"""
    def __init__(self, data, args, knowledge_file):
        super().__init__(data, args)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = knowledge_file
        self.kenn_layers = torch.nn.ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=knowledge_file))

    def reset_parameters(self):
        super().reset_parameters()  # should call reset parameter function of MLP
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight)

        # call kenn-sub layers
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)

        # return F.softmax(z, dim=-1)
        return z.log_softmax(dim=-1)


class KENN_MLP(MLP):
    """ kenn-sub with MLP (from ogb) as base NN"""

    def __init__(self, data, args, knowledge_file):
        super().__init__(data, args)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = knowledge_file
        self.kenn_layers = torch.nn.ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=knowledge_file))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations=None, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight)

        # call kenn-sub layers
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)
        # return F.softmax(z, dim=-1)
        return z.log_softmax(dim=-1)


class KENN_SAGE(SAGE):
    """ kenn-sub with GraphSage (from ogb) as base NN"""
    def __init__(self, data, args, knowledge_file):
        super().__init__(data, args)
        self.name = str('KENN_' + self.name)
        self.knowledge_file = knowledge_file
        self.kenn_layers = torch.nn.ModuleList()

        for _ in range(args.num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=knowledge_file))

    def reset_parameters(self):
        super().reset_parameters()  # should call reset parameter function of MLP
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, relations, edge_weight=None):
        z = super().forward(x, edge_index, relations, edge_weight)

        # call kenn-sub layers
        for layer in self.kenn_layers:
            z, _ = layer(unary=z, edge_index=edge_index, binary=relations)

        return z.log_softmax(dim=-1)
