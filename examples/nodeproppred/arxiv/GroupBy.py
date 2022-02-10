# TODO: rewrite to Torch

import torch
from torch_scatter import scatter_add


class GroupBy(torch.nn.Module):
    def __init__(self, number_of_unary_predicates):
        super(GroupBy, self).__init__()
        self.n_unary = number_of_unary_predicates

    def reset_parameters(self):
        """ is not needed here I think """

    def forward(self, unary, binary, deltas, index1, index2):
        """Split the deltas matrix in unary and binary deltas.
        #todo: why do we need binary here
        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param deltas: the tensor containing the delta values
        :param index1: a vector containing the indices of the first object
        of the pair referred by binary and deltas tensors
        :param index2: a vector containing the indices of the second object
        of the pair referred by binary and deltas tensors
        """

        ux = deltas[:, :self.n_unary]
        uy = deltas[:, self.n_unary:2 * self.n_unary]
        b = deltas[:, 2 * self.n_unary:]

        ux_deltas = torch.zeros(unary.shape).scatter_(dim=0, index=torch.unsqueeze(index1, 1), src=ux, reduce='add')
        uy_deltas = torch.zeros(unary.shape).scatter_(dim=0, index=torch.unsqueeze(index2, 1), src=uy, reduce='add')

        assert ux_deltas.shape == uy_deltas.shape, 'GroupBy: deltas for ux and uy must have the same shape'

        return torch.add(ux_deltas, uy_deltas), b
