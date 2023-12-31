import torch
from kegnn.ClauseEnhancer import ClauseEnhancer
from kegnn.boost_functions import *
from torch_scatter import scatter_add


class KnowledgeEnhancer(torch.nn.Module):

    def __init__(self, predicates: [str], clauses: [str], initial_clause_weight=0.5, save_training_data=False,
                 boost_function=GodelBoostConormApprox):
        """Initialize the knowledge base.
        :param predicates: a list of predicates names
        :param clauses: a list of constraints. Each constraint is a string on the form:
        clause_weight:clause
        The clause_weight should be either a real number (in such a case this value is fixed) or an underscore
        (in this case the weight will be a tensorflow variable and learned during training).
        The clause must be represented as a list of literals separated by commas (that represent disjunctions).
        Negation must be pecified by adding the letter 'n' before the predicate name.
        An example:
           _:nDog,Animal
        :param initial_clause_weight: the initial sign to the clause weight. Used if the clause weight is learned.
        """

        super().__init__()
        self.clause_enhancers = []
        self.save_training_data = save_training_data

        # todo: different modalities to define clause weight
        if initial_clause_weight == "random":
            self.initial_clause_weight = torch.rand(len(clauses))
        else:
            self.initial_clause_weight = initial_clause_weight * torch.ones(len(clauses))

        for index, clause in enumerate(clauses):
            enhancer = ClauseEnhancer(
                predicates, clause[:-1], self.initial_clause_weight[index].item(), boost_function=boost_function)
            self.clause_enhancers.append(enhancer)
            self.add_module(f'clause-{index}', enhancer)

    def forward(self, ground_atoms: torch.Tensor, using_max=False) -> (torch.Tensor, [torch.Tensor, torch.Tensor]):
        """Improve the satisfaction level of a set of clauses.
        :param ground_atoms: the tensor containing the pre-activation values of the ground atoms
        :return: final delta values"""

        # scatter_deltas_list will be the list of deltas for each clause
        # e.g. scatter_deltas_list[0] are the deltas relative to the first clause.
        scatter_deltas_list: [torch.Tensor] = []
        light_deltas_list = []
        weights = []
        deltas_list = []
        indexes_list = []
        # TODO: parllelize over clauses
        for enhancer in self.clause_enhancers:
            #print(torch.cuda.memory_summary())
            #scattered_delta, delta = enhancer(ground_atoms)
            delta, indices = enhancer(ground_atoms)
            deltas_list.append(delta)
            indexes_list.append(indices)
            #scatter_deltas_list.append(scattered_delta)
            if self.save_training_data:
                light_deltas_list.append(delta)
                weights.append(enhancer.clause_weight.numpy()[0][0])

        deltas_data = [light_deltas_list, weights]

        all_deltas = torch.cat(deltas_list, dim=1)
        all_indices = torch.cat(indexes_list, dim=0)


        # The sum can be refactored into the for loop above.
        if using_max:
            # TODO: the max is not performed at the level of groupby (sum is still used there)
            stacked_deltas = torch.stack(scatter_deltas_list)
            _, indexes = torch.abs(stacked_deltas).max(dim=0)
            return torch.gather(stacked_deltas, 0, indexes.unsqueeze(0)), deltas_data
        else:
            result = scatter_add(src=torch.transpose(all_deltas, 0, 1), index=all_indices, dim=0)
            return torch.transpose(result, 0, 1), deltas_data
