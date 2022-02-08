# TODO: rewrite to Torch

import torch


class GroupBy(torch.nn.Module):
    def __init__(self, number_of_unary_predicates):
        super(GroupBy, self).__init__()
        self.n_unary = number_of_unary_predicates

    def reset_parameters(self):
        """ is not needed here I think """

    def forward(self, unary, binary, deltas, index1, index2):
        # todo: see if this runs + check dimensions
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
        shape = unary.size()
        #shape = tf.cast(tf.shape(unary), dtype=tf.int64)

        #inputs = tf.placeholder(tf.int32, shape)

        #return tf.scatter_nd(index1, ux, shape) + tf.scatter_nd(index2, uy, shape), b
        #return tf.scatter_nd(tf.expand_dims(index1, axis=1), ux, shape) + tf.scatter_nd(tf.expand_dims(index2, axis=1), uy, shape), b
        return torch.scatter(src=torch.unsqueeze(index1, dim=1), dim=shape, index=ux) + torch.scatter(src=torch.unsqueeze(index2, dim=1), dim=shape, index=uy), b
