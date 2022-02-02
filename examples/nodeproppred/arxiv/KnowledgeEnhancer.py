# TODO: rewrite to Torch

"""Write here the implementation of KnowledgeEnhancer.py"""
import torch
import ClauseEnhancer


class KnowledgeEnhancer(torch.nn.Module):

    def __init__(self, predicates, clauses, initial_clause_weight=0.5, save_training_data=False, **kwargs):
        """Initialize the knowledge base.

        :param predicates: a list of predicates names
        :param clauses: a list of constraints. Each constraint is a string on the form:
        clause_weight:clause

        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).

        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must specified by adding the letter 'n' before the predicate name.

        An example:
           _:nDog,Animal

        :param initial_clause_weight: the initial sign to the clause weight. Used if the clause weight is learned.

        """

        super(KnowledgeEnhancer, self).__init__()
        self.predicates = predicates
        self.clauses = clauses
        self.initial_clause_weight = initial_clause_weight
        self.clause_enhancers = []
        self.save_training_data = save_training_data
        self.clause_enhancers = torch.nn.ModuleList()
        for clause in self.clauses:
            self.clause_enhancers.append(ClauseEnhancer(
                self.predicates, clause[:-1], self.initial_clause_weight, self.save_training_data))

    def forward(self, inputs, **kwargs):
        """Improve the satisfaction level of a set of clauses.

        :param inputs: the tensor containing predicates' pre-activation values for many entities
        :return: final delta values"""

        # scatter_deltas_list will be the list of deltas for each clause
        # e.g. scatter_deltas_list[0] are the deltas relative to the first clause.

        deltas_list = []
        indexes_list = []

        for clause in self.clause_enhancers:
            delta, indexes = clause(inputs)

            deltas_list.append(delta)
            indexes_list.append(indexes)

        all_deltas = torch.cat(deltas_list, dim=1)
        all_indexes = torch.cat(indexes_list, dim=0)

        #todo - check what this does
        return torch.transpose(
            torch.scatter_add(all_indexes, torch.transpose(all_deltas), torch.flip(torch.size(inputs), [0])))