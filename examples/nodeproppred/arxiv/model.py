""" Put here all the defined models for KENN/Base NN """
"""
-MLP: ogb baseline
-GCN: ogb baseline
-GraphSAGE: ogb baseline 
-Standard: base neural network architecture propoesd by KENN - still strange results
--> I would suggest not to use this as a baseline anymore but the ogb baselines 

"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from examples.nodeproppred.arxiv.parsers import *


class GCN(torch.nn.Module):
    """
    GCN module baseline given by OGB
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.name = 'GCN'
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, relations):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    """ Implementation of GraphSAGE - ogb baseline  """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()
        self.name = 'SAGE'
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, relations):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class MLP(torch.nn.Module):
    """ MLP baseline for OGB """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        self.name = 'MLP'
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    #def forward(self, x, adj_t=None, relations=None):  # The None is needed for KENN heritage
    #def forward(self, x, adj_t, relations):
    def forward(self, x, adj_t, relations):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


class Standard(torch.nn.Module):
    """ Original Base Neural Network by KENN paper """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(Standard, self).__init__()
        self.lin_layers = torch.nn.ModuleList()
        self.lin_layers.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lin_layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lin_layers.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for layer in self.lin_layers:
            layer.reset_parameters()

    def forward(self, x, adj_t):
        for i, lin in enumerate(self.lin_layers[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_layers[-1](x)
        return torch.softmax(x, dim=-1)


class KENN(MLP):
    """ KENN with MLP (from ogb) as base NN"""

    def __init__(self, knowledge_file, hidden_channels, in_channels, out_channels, num_layers, num_kenn_layers, dropout, relations,
                 explainer_object=None):
        super(KENN, self).__init__(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                                   num_layers=num_layers, dropout=dropout)
        self.name = 'KENN_MLP' #str('KENN_' + super(KENN, self).name)
        self.knowledge_file = knowledge_file
        self.explainer_object = explainer_object
        self.kenn_layers = torch.nn.ModuleList()

        for _ in range(num_kenn_layers):
            self.kenn_layers.append(relational_parser(knowledge_file=knowledge_file, explainer_object=explainer_object))

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.kenn_layers:
            layer.reset_parameters()

    def forward(self, x, adj_t, relations):
        z = super(KENN, self).forward(x)

        # call KENN layers
        for layer in self.kenn_layers:

            z = layer(z, adj_t, relations)

        return torch.softmax(z)
