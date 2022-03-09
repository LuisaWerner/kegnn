import torch
from torch import mul
from torch.nn.functional import softmax


class ClauseEnhancer(torch.nn.Module):
    """Clause Enhancer layer
    """

    def __init__(self, available_predicates, clause_string, initial_clause_weight, input_shape,
                 save_training_data=False):
        """Initialize the clause.
        :param available_predicates: the list of all possible literals in a clause
        :param clause_string: a string representing a conjunction of literals. The format should be:
        clause_weight:clause
        The clause_weight should be either a real number (in such a case this sign is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).
        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.
        An example:
           _:nDog,Animal
        :param initial_clause_weight: the initial sign to the clause weight. Used if the clause weight is learned.
        """

        super(ClauseEnhancer, self).__init__()
        string = clause_string.split(':')
        self.original_string = string[1]
        self.string = string[1].replace(
            ',', 'v').replace('(', '').replace(')', '')

        if string[0] == '_':
            self.initial_weight = initial_clause_weight
            self.hard_clause = False
        else:
            self.initial_weight = int(string[0])
            self.hard_clause = True

        self.hard_clause = string[0] != '_'

        literals_list = string[1].split(',')
        self.number_of_literals = len(literals_list)

        self.gather_literal_indices = []
        self.scatter_literal_indices = []
        signs = []

        for literal in literals_list:
            sign = 1
            if literal[0] == 'n':
                sign = -1
                literal = literal[1:]

            literal_index = available_predicates.index(literal)
            self.gather_literal_indices.append(literal_index)
            self.scatter_literal_indices.append([literal_index])
            signs.append(sign)

        self.signs = torch.Tensor(signs)
        self.save_training_data = save_training_data
        self.cuda()

        """self.clause_weight = self.add_weight(
            name='kernel',
            shape=(1, 1),
            initializer=tf.keras.initializers.Constant(
                value=self.initial_weight),
            constraint=RangeConstraint(),
            trainable=not self.hard_clause)
        """
        # todo self.clause_weight = [...]
        self.register_parameter(name="clause_weight",
                                param=torch.nn.Parameter(torch.Tensor([self.initial_weight]),
                                                         requires_grad=not self.hard_clause))

    def reset_parameters(self):
        # todo: check if this really works as it should
        """ resets clause weights after one iteration back to the initial clause weight """
        self.clause_weight = torch.nn.Parameter(torch.Tensor([self.initial_weight]), requires_grad=not self.hard_clause)

    def grounded_clause(self, inputs):
        """Find the grounding of the clause
        :param inputs: the tensor containing predicates' pre activations for many objects
        :return: the grounded clause (a tensor with literals truth values)
        """
        selected_predicates = inputs[:, self.gather_literal_indices]
        print(f'ClauseEnhancer: selected predicates on cuda? {selected_predicates.is_cuda}')
        print(f'ClauseEnhancer: self.signs on cuda? {self.signs.is_cuda}')
        clause_matrix = mul(selected_predicates, self.signs)

        return clause_matrix

    def forward(self, inputs, **kwargs):
        """Improve the satisfaction level of the clause.
        :param inputs: the tensor containing predicates' pre-activation values for many entities
        :return: delta vector to be summed to the original pre-activation tensor to obtain an higher satisfaction of \
        the clause"""

        clause_matrix = self.grounded_clause(inputs)
        print(
            f'Clause Enhancer (forward) clause_matrix, self.signs on cuda? {clause_matrix.is_cuda, self.signs.is_cuda}')

        delta = self.signs * softmax(clause_matrix, dim=-1) * self.clause_weight

        return delta, torch.Tensor(self.scatter_literal_indices).to(torch.int64)
