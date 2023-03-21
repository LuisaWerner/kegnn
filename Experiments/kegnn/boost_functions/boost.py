import abc

import numpy as np
import torch
from torch.nn.functional import softmax


class BoostFunction(torch.nn.Module, abc.ABC):

    def __init__(self, initial_weight: float, fixed_weight: bool,
                 min_weight, max_weight):
        super().__init__()
        self.initial_weight = initial_weight
        self.register_parameter(
            name='clause_weight',
            param=torch.nn.Parameter(torch.tensor(initial_weight)))

        self.clause_weight.requires_grad = not fixed_weight
        self.min_weight = min_weight
        self.max_weight = max_weight

    def forward(self, selected_predicates: torch.Tensor, signs: torch.Tensor):
        """
        :param selected_predicates: [b, l] The pre-activations of the selected ground atoms.
        :param signs: [l] The signs of the literals
        :return: [b, l] The delta given by the boost function
        """
        pass

    def reset_parameters(self):
        torch.nn.init.constant_(self.clause_weight, self.initial_weight)


class GodelBoostConormApprox(BoostFunction):
    def __init__(self, initial_weight: float, fixed_weight: bool, min_weight, max_weight):
        super().__init__(initial_weight, fixed_weight, min_weight, max_weight)

    def forward(self, selected_predicates: torch.Tensor, signs: torch.Tensor):
        self.clause_weight.data = torch.clip(self.clause_weight, self.min_weight, self.max_weight)

        clause_matrix = selected_predicates * signs

        # Approximated Godel t-conorm boost function on preactivations
        return signs * softmax(clause_matrix, dim=-1) * self.clause_weight



