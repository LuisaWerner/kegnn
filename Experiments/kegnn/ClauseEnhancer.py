import torch
from kegnn.boost_functions import *
import importlib


class ClauseEnhancer(torch.nn.Module):

    def __init__(self,
                 available_predicates: [str],
                 clause_string: str,
                 initial_clause_weight: float,
                 min_weight=0,
                 max_weight=500,
                 boost_function=GodelBoostConormApprox):
        """Initialize the clause.
        :param available_predicates: the list of all possible literals in a clause
        :param clause_string: a string representing a conjunction of literals. The format should be:
        clause_weight:clause
        The clause_weight should be either a real number (in such a case this sign is fixed) or an underscore
        (in this case the weight will be learned during training).
        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must be specified by adding the letter 'n' before the predicate name.
        An example:
           _:nDog,Animal
        :param initial_clause_weight: the initial value of the clause weight. Used if the clause weight is learned.
        """
        super().__init__()
        boost_module = importlib.import_module("kenn.boost_functions")
        _boost_function = getattr(boost_module, boost_function)
        # Split weight and clause
        weight_clause_split = clause_string.split(':')

        self.clause_string = weight_clause_split[1]
        weight_string = weight_clause_split[0]
        self.string = weight_clause_split[1].replace(
            ',', 'v').replace('(', '').replace(')', '')

        if weight_string == '_':
            initial_weight = initial_clause_weight
            fixed_weight = False
        else:
            initial_weight = float(weight_string)
            fixed_weight = True

        self.conorm_boost = _boost_function(initial_weight, fixed_weight, min_weight, max_weight)

        literals = self.clause_string.split(',')
        self.number_of_literals = len(literals)

        # Setup indexing of the literals
        gather_literal_indices = []
        scatter_literal_indices = []
        signs = []

        for literal in literals:
            sign = 1
            if literal[0] == 'n':
                sign = -1
                literal = literal[1:]

            literal_index = available_predicates.index(literal)
            gather_literal_indices.append(literal_index)
            scatter_literal_indices.append([literal_index])
            signs.append(sign)

        self.register_buffer('signs', torch.tensor(signs, dtype=torch.float32))
        self.register_buffer('gather_literal_indices', torch.Tensor(gather_literal_indices).to(torch.long))
        self.register_buffer('scatter_literal_indices', torch.Tensor(scatter_literal_indices).to(torch.long))

    def select_predicates(self, ground_atoms: torch.Tensor) -> torch.Tensor:
        """Find the grounding of the clause
        :param ground_atoms: [b, 2|U| + |B|] the tensor containing the pre activations of the ground atoms
        :return: the grounded clause: [b, l] (a tensor with literals pre-activations)
        """

        # Choose the right literals
        # self.gather_literal_indices: [l], each in {1, ..., 2|U| + |B|}
        return ground_atoms[..., self.gather_literal_indices]

    def forward(self, ground_atoms: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Improve the satisfaction level of the clause.
        :param ground_atoms: the tensor containing the pre-activation values of the ground atoms
        :return: delta vector to be summed to the original pre-activation tensor to obtain an higher satisfaction of \
        the clause"""
        # [b, l]
        selected_predicates = self.select_predicates(ground_atoms)

        delta = self.conorm_boost(selected_predicates, self.signs)

        # [b, 2|U|+|B|]
        scattered_delta = torch.zeros_like(ground_atoms)
        scattered_delta[..., self.gather_literal_indices] = delta

        return delta, self.scatter_literal_indices
