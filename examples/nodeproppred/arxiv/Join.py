import numpy
import torch
from torch import reshape


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
        :param index2: a vector containing the indices of the second object
        of the pair referred by binary tensor
        output corresponds to matrix M in the paper.
        """

        if index1.dim() == 0 and index2.dim() == 0:
            index1 = reshape(index1, (1,))
            index2 = reshape(index2, (1,))

        # returns matrix M of paper
        a = unary[index1]
        b = unary[index2]

        return numpy.concatenate([a, b, binary], dim=1)

        # return torch.cat([unary[index1].detach(), unary[index2].detach(), binary.detach()], dim=1) # todo is detach okay
