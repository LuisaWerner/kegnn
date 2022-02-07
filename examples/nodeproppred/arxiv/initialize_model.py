import torch
from model import *


def initialize(args, data):
    """
    initializes the model
    args: input arguments
    data: a data object with x, y, and adjacency matrix (if relational object)
    """
    if args.model == 'MLP':
        if args.use_node_embedding:
            embedding = torch.load('embedding.pt', map_location='cpu')
            data.x = torch.cat([data.x, embedding], dim=-1)
        model = MLP(data.x.size(-1), args.hidden_channels, data.num_classes,
                    args.num_layers, args.dropout)

    elif args.model == 'Standard':
        # set parameters for Standard NN of KENN
        args.use_node_embedding = False
        args.num_layers = 4
        args.hidden_channels = 50
        args.dropout = 0.5
        args.lr = 0.01
        args.epochs = 300
        args.runs = 500

        if args.use_node_embedding:
            embedding = torch.load('embedding.pt', map_location='cpu')
            data.x = torch.cat([data.x, embedding], dim=-1)

        model = Standard(data.x.size(-1), args.hidden_channels, data.num_classes,
                         args.num_layers, args.dropout)

    elif args.model == 'SAGE':
        model = SAGE(data.num_features, args.hidden_channels,
                     data.num_classes, args.num_layers,
                     args.dropout)

    elif args.model == 'GCN':
        model = GCN(data.num_features, args.hidden_channels,
                    data.num_classes, args.num_layers,
                    args.dropout)
    else:
        print('unknown model :', args.model)

    return model
