# TODO: check if this runs

import torch

from KENN.KnowledgeEnhancer import KnowledgeEnhancer


class Kenn(torch.nn.Module):

    def __init__(self, predicates, clauses, activation=lambda x: x, initial_clause_weight=0.5, save_training_data=False,
                 **kwargs):
        super().__init__()
        self.predicates = predicates
        self.clauses = clauses
        self.activation = activation
        self.initial_clause_weight = initial_clause_weight
        self.save_training_data = save_training_data
        self.knowledge_enhancer = KnowledgeEnhancer(
        self.predicates, self.clauses, self.initial_clause_weight, self.save_training_data)

    def reset_parameters(self):
        pass

    def forward(self, inputs):
        """Improve the satisfaction level of a set of clauses.

        :param inputs: the tensor containing predicates' pre-activation values for many entities
        :return: final preactivations"""

        if self.save_training_data:
            deltas, deltas_list = self.knowledge_enhancer(inputs)
            return self.activation(inputs + deltas), deltas_list
        else:
            deltas = self.knowledge_enhancer(inputs)
            return self.activation(inputs + deltas)

    def get_config(self):
        config = super(Kenn, self).get_config()
        config.update({'predicates': self.predicates})
        config.update({'clauses': self.clauses})
        config.update({'activation': self.activation})
        config.update({'initial_clause_weight': self.initial_clause_weight})
        # config['output_size'] =  # say self. _output_size  if you store the argument in __init__
        return config