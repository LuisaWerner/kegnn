

import torch
from torch import cat, reshape


class Join(torch.nn.Module):
    """Join layer

    """

    def __init__(self):
        super(Join, self).__init__()

    def forward(self, unary, binary, index1, index2):
        # todo: see if this runs
        """Join the unary and binary tensors.

        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param index1: a vector containing the indices of the first object
        of the pair referred by binary tensor
        :param index1: a vector containing the indices of the second object
        of the pair referred by binary tensor
        output corresponds to matrix M in the paper.
        """

        if index1.dim() == 0 and index2.dim() == 0:
            index1 = reshape(index1, (1,))
            index2 = reshape(index2, (1,))

        # returns matrix M of paper
        return cat([unary[index1], unary[index2], binary], dim=1)
