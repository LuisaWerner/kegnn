# Todo: verify if this does what it should
# Corresponds to keras Range Constraint
import torch


class WeightClipper(object):
    """  makes sure that the clause weights remain in a given range """

    def __init__(self, lower=0, upper=500):
        self.lower = lower
        self.upper = upper

    def __call__(self, module):
        if hasattr(module, 'clause_weight'):
            w = module.weight.data
            w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w))

""" todo
call it later with
clipper = WeightClipper(0, 500)
after optimization step: optimizer.step()
model.apply(clipper)"""