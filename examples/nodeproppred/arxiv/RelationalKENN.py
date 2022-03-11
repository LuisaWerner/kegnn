import torch

from GroupBy import GroupBy
from Join import Join
from KnowledgeEnhancer import KnowledgeEnhancer


class RelationalKENN(torch.nn.Module):

    def __init__(self,
                 unary_predicates,
                 binary_predicates,
                 unary_clauses,
                 binary_clauses,
                 activation=lambda x: x,
                 initial_clause_weight=0.5):

        """Initialize the knowledge base.
        :param input_shape: size of the input
        :param unary_predicates: the list of unary predicates names
        :param binary_predicates: the list of binary predicates names
        :param unary_clauses: a list of unary clauses. Each clause is a string on the form:
        clause_weight:clause

        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).

        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.

        An example:
           _:nDog,Animal

        :param binary_clauses: a list of binary clauses
        :param activation: activation function
        :param initial_clause_weight: initial value for the cluase weight (if clause is not hard)
        """

        super().__init__()

        self.unary_predicates = unary_predicates
        self.n_unary = len(unary_predicates)
        self.unary_clauses = unary_clauses
        self.binary_predicates = binary_predicates
        self.binary_clauses = binary_clauses
        self.activation = activation
        self.initial_clause_weight = initial_clause_weight

        self.unary_ke = None
        self.binary_ke = None
        self.join = None
        self.group_by = None

        # BUILD THE MODEL
        if len(self.unary_clauses) != 0:
            self.unary_ke = KnowledgeEnhancer(
                self.unary_predicates, self.unary_clauses, initial_clause_weight=self.initial_clause_weight)
        if len(self.binary_clauses) != 0:
            self.binary_ke = KnowledgeEnhancer(
                self.binary_predicates, self.binary_clauses, initial_clause_weight=self.initial_clause_weight)

            self.join = Join()
            self.group_by = GroupBy(self.n_unary)

    def reset_parameters(self):
        """Relational KENN and KnowledgeEnhancer don't have parameters
        Resetting parameters of the knowledge enhancement layer means
        resetting clause weights to initial clause weight """

        if len(self.unary_clauses) != 0:
            for ce in self.unary_ke.clause_enhancers:
                ce.reset_parameters()

        if len(self.binary_clauses) != 0:
            for ce in self.binary_ke.clause_enhancers:
                ce.reset_parameters()

    def forward(self, unary, binary, edge_index):
        """Forward step of Kenn model for relational data.
        :param unary: the tensor with unary predicates pre-activations
        :param binary: the tensor with binary predicates pre-activations
        :param edge_index: describes the first and second component of the link
        """

        index1 = edge_index[0]
        index2 = edge_index[1]

        if len(self.unary_clauses) != 0:
            deltas_sum, deltas_u_list = self.unary_ke(unary)
            u = unary + deltas_sum
        else:
            u = unary
            deltas_u_list = torch.unsqueeze(input=torch.zeros(unary.shape), dim=0)

        if len(self.binary_clauses) != 0 and len(binary) != 0:
            joined_matrix = self.join(u, binary, index1, index2)
            deltas_sum = self.binary_ke(joined_matrix)

            delta_up, delta_bp = self.group_by(
                u, binary, deltas_sum, index1, index2)
        else:
            delta_up = torch.zeros(u.shape)
            delta_bp = torch.zeros(binary.shape)

        return self.activation(u + delta_up), self.activation(binary + delta_bp) # todo: slow

    def get_config(self):
        config = super(RelationalKENN, self).get_config()
        config.update({'unary_predicates': self.unary_predicates})
        config.update({'unary_clauses': self.unary_clauses})
        config.update({'binary_predicates': self.binary_predicates})
        config.update({'binary_clauses': self.binary_clauses})
        config.update({'activation': self.activation})
        config.update({'initial_clause_weight': self.initial_clause_weight})

        return config



