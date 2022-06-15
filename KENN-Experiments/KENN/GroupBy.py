
import torch
from torch_scatter import scatter_add


class GroupBy(torch.nn.Module):
    def __init__(self, number_of_unary_predicates):
        super().__init__()
        self.n_unary = number_of_unary_predicates
        self.register_buffer('unary_zeroes', torch.zeros(1, 1))

    def reset_parameters(self):
        """ no trainable parameters - not needed  """
        pass

    def forward(self, unary, binary, deltas, index1, index2):
        """Split the deltas matrix in unary and binary deltas.
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

        uy_deltas = scatter_add(src=uy, index=torch.unsqueeze(index2, 1), dim=0,
                                out=self.unary_zeroes.repeat(unary.shape))  # out=torch.zeros(unary.shape))
        ux_deltas = scatter_add(src=ux, index=torch.unsqueeze(index1, 1), dim=0,
                                out=self.unary_zeroes.repeat(unary.shape))  # cuda

        assert ux_deltas.shape == uy_deltas.shape, 'GroupBy: deltas for ux and uy must have the same shape'

        return torch.add(ux_deltas, uy_deltas), b
