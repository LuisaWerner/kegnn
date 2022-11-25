import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def clause_compliance(data: Data, clause: int):
    """ Calculates the clause_compliance
    Clause compliance = # neighbors that both have the respective class cls / # total num neighbors
    @param data: PyG Data Object
    @param clause: class [here defined as integer]
    """
    y = data.y.cpu().detach().numpy()
    edge_index = np.transpose(data.edge_index.cpu().detach().numpy())

    cls_mask = np.where(y == clause)[0]  # the indexes with cls
    cls_list = []
    for node in cls_mask:
        edge_mask = edge_index[
                    np.where(np.logical_or([edge_index[:, 0] == node][0], [edge_index[:, 1] == node][0])), :]
        cls_list += [np.take(y, edge_mask).squeeze(axis=0)]

    cls_list = np.concatenate(cls_list, axis=0)
    n_neighbors = len(cls_list)
    n_neighbors_equal = len(np.where(cls_list[:, 0] == cls_list[:, 1])[0])

    return n_neighbors_equal / n_neighbors


class KnowledgeGenerator(object):
    # todo calculate clause compliance on original set or after inductive/drop_edges preprocessing
    def __init__(self, model, args):
        super(KnowledgeGenerator, self).__init__()
        self.train_data = model.train_data
        self.data = model.train_data
        self.cc_range = args.cc_range
        self.cls_range = args.cls_range  # todo add args parameters
        self.keys = ['all', 'train', 'val', 'test']
        self._compliance = {}
        self._quantity = {}

    def __setattr__(self, key, value):
        if 'threshold' in key and value < 0:
            raise ValueError(f' No negative numbers allowed for {key}')
        super().__setattr__(key, value)

    def compute_clause_stats(self):
        """ this returns a nested dictionary self._compliance[cls][mask][value] """
        for key in self.keys:
            if key == 'all':
                mask = key
                split_data = self.data.y.cpu().detach().numpy()

            else:
                mask = key + '_mask'
                assert hasattr(self.data, mask)
                split_data = Data(x=self.data.x[getattr(self.data, mask)], y=self.data.y[getattr(self.data, mask)],
                                  edge_index=
                                  subgraph(getattr(self.data, mask), self.data.edge_index, relabel_nodes=True)[0])

            for cls in range(self.data.num_classes):
                # compute clause compliance per clause id
                self._compliance[cls] = {mask: clause_compliance(split_data, cls)}  # todo check

                # compute quantity per clause id
                y = split_data.y.cpu().detach().numpy()
                self._quantity[cls] = {mask: np.unique(y, return_counts=True).sort(0)[1]/split_data.num_nodes}


    def generate_knowledge(self, args):
        """
        creates the knowledge file based on unary predicates = document classes
        cite is binary predicate
        num_classes int
        """
        # todo more generic formulation for Cite, check parser that it works
        # define which thresholds should be respected and put thresholds
        assert hasattr(self.data, 'num_classes')

        if args.create_kb:
            class_list = list(range(self.data.num_classes))
            # class_list = []
            for i in class_list:  # if quantity of class
                class_list[i] = 'class_' + str(i)
            # Generate knowledge
            kb = ''

            # List of predicates
            for c in class_list:
                kb += c + ','

            kb = kb[:-1] + '\nCite\n\n'

            # No unary clauses

            kb = kb[:-1] + '\n>\n'

            # Binary clauses

            # nC(x),nCite(x.y),C(y)
            for c in class_list:
                kb += '_:n' + c + '(x),nCite(x.y),' + c + '(y)\n'

            with open('knowledge_base', 'w') as kb_file:
                kb_file.write(kb)

            return kb

        else:
            # use the kb defined in args
            with open('knowledge_base', 'w') as kb_file:
                kb_file.write(args.knowledge_base)
            return args.knowledge_base


# todo write main method to test