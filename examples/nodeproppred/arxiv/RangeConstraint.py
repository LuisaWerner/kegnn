import torch


class RangeConstraint(object):
    """
    makes sure that the clause_weights remain in a given range
    Corresponds to keras Range Constraint
    """

    def __init__(self, lower=0, upper=500):
        self.lower = lower
        self.upper = upper

    def __call__(self, module):
        if hasattr(module, 'clause_weight'):
            w = module.clause_weight.data
            module.clause_weight = torch.nn.Parameter(torch.clamp(w, min=self.lower, max=self.upper))
            return torch.clamp(w, min=self.lower, max=self.upper)
