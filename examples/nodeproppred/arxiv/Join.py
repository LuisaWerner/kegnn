# TODO: rewrite to Torch

import torch


class Join(torch.nn.Module):
    """Join layer

    """

    def __init__(self):
        super(Join, self).__init__()

    def call(self, unary, binary, index1, index2):
        """Join the unary and binary tensors.

        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param index1: a vector containing the indices of the first object
        of the pair referred by binary tensor
        :param index1: a vector containing the indices of the second object
        of the pair referred by binary tensor
        output corresponds to matrix M in the paper.
        """

        index1 = torch.squeeze(index1)
        index2 = torch.squeeze(index2)

        # For the case where index1 and index2 were of length 1, tf.squeeze will make their rank = 0
        if index1.shape.rank == 0 and index2.shape.rank == 0:
            index1 = torch.reshape(index1, (1,))
            index2 = torch.reshape(index2, (1,))

        # returns matrix M of paper
        # todo: slow
        return torch.cat([torch.gather(unary, index1, axis=0), torch.gather(unary, index2, axis=0), binary], axis=1)
