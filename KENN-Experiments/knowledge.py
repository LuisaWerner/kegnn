import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import pathlib
import warnings
from argparse import ArgumentError


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
        # get the relations that include the node
        edge_mask = edge_index[
                    np.where(np.logical_or([edge_index[:, 0] == node][0], [edge_index[:, 1] == node][0])), :]
        # get classes for node indices in pair
        cls_list += [np.take(y, edge_mask).squeeze(axis=0)]

    cls_list = np.concatenate(cls_list, axis=0)
    n_neighbors = len(cls_list)

    if n_neighbors == 0:
        warnings.warn('RuntimeWarning: No neighbors contained in Clause Compliance Calculation. Might be because of an empty set. Verify edges_drop_rate argument. Clause Compliance set to 0.0 for this case ')
        return 0.0

    n_neighbors_equal = len(np.where(cls_list[:, 0] == cls_list[:, 1])[0])
    return n_neighbors_equal / n_neighbors


class ClauseStats(object):
    """ for each clause compliance: dict ['all', 'train', ...]: value, store stats"""

    def __init__(self):
        super(ClauseStats, self).__init__()
        self.keys = ['all', 'train', 'val', 'test']
        self.compliance = dict.fromkeys(self.keys)
        self.quantity = dict.fromkeys(self.keys)


class KnowledgeGenerator(object):
    """ class to treat the knowledge generation """

    def __init__(self, model, args):
        super(KnowledgeGenerator, self).__init__()
        self.keys = ['all', 'train', 'val', 'test']
        self.train_data = model.train_data
        self.data = model.train_data
        self.dataset = args.dataset
        self.create_kb = args.create_kb
        self.knowledge_base = args.knowledge_base
        self.filter_key = args.knowledge_filter_key
        self.compliance_range = args.compliance_range
        self.quantity_range = args.quantity_range
        self.clause_stats = []

        self.delete_files()
        self.compute_clause_stats()

        if args.save_data_stats and not pathlib.Path(f'{self.dataset}_data_stats').exists():
            print('Saving Data Stats..... ')
            self.save_data_stats()

    @property
    def knowledge(self):
        self.generate_knowledge()
        return f'{self.dataset}_knowledge_base'

    def delete_files(self):
        """ Deletes knowledge base and datastats file that might
        still be in directory from previous runs """
        know_base = pathlib.Path(f'{self.dataset}_knowledge_base')
        stats = pathlib.Path(f'{self.dataset}_stats')
        if know_base.is_file():
            know_base.unlink()
            print(f'{self.dataset} knowledge base deleted')
        if stats.is_file():
            stats.unlink()
            print(f'{self.dataset} stats deleted')

    def filter_clause(self, clause):
        """ filters clauses according to quantity and compliance ranges """
        # clauses filtered by corresponding int
        compliance = self.clause_stats[clause].compliance[self.filter_key]
        quantity = self.clause_stats[clause].quantity[self.filter_key]
        if self.compliance_range[0] <= compliance <= self.compliance_range[1] \
                and self.quantity_range[0] <= quantity <= self.quantity_range[1]:
            return True
        else:
            return False

    def compute_clause_stats(self):
        """ computes quantity and clause compliance for each clause for keys train, val, test and overall"""
        print('Compute clause stats...')
        for cls in range(self.data.num_classes):
            c = ClauseStats()
            for key in self.keys:
                if key == 'all':
                    split_data = self.data
                else:
                    mask = key + '_mask'
                    assert hasattr(self.data, mask)
                    split_data = Data(x=self.data.x[getattr(self.data, mask)], y=self.data.y[getattr(self.data, mask)],
                                      edge_index=
                                      subgraph(getattr(self.data, mask), self.data.edge_index, relabel_nodes=True)[0])

                c.compliance[key] = clause_compliance(split_data, cls)
                cnt = np.asarray(np.unique(split_data.y.cpu().detach().numpy(), return_counts=True))
                try:
                    cnt = cnt[cnt[:, 1].argsort()][1, cls]
                except IndexError:
                    for x in range(0, self.data.num_classes - 1):
                        if x not in cnt[0]:
                            cnt = np.insert(cnt, 0, values=np.zeros(shape=(2,)), axis=1)
                            cnt = cnt[cnt[:, 1].argsort()][1, cls]
                c.quantity[key] = cnt / split_data.num_nodes
            self.clause_stats += [c]  # for each clause one object

    def generate_knowledge(self):
        """
        creates the knowledge file based on unary predicates = document classes
        cite is binary predicate
        num_classes int
        """
        assert hasattr(self.data, 'num_classes')

        if self.create_kb:
            # Filter clauses
            filtered_clauses = list(filter(lambda x: self.filter_clause(x), list(range(self.data.num_classes))))
            class_list = []
            for i in filtered_clauses:  # if quantity of class
                class_list += ['class_' + str(i)]

            if not class_list:
                UserWarning('Empty knowledge base. Choose other filters to keep more clauses ')
                return ''

            # Generate knowledge
            kb = ''

            # List of predicates
            for c in class_list:
                kb += c + ','
            # for c in range(self.data.num_classes):
            #     kb += 'class_' + str(c)+ ', '

            kb = kb[:-1] + '\nLink\n\n'

            # No unary clauses

            kb = kb[:-1] + '\n>\n'

            # Binary clauses

            # eg: nC(x),nCite(x.y),C(y)
            for c in class_list:
                kb += '_:n' + c + '(x),nLink(x.y),' + c + '(y)\n'

            with open(f'{self.dataset}_knowledge_base', 'w') as kb_file:
                kb_file.write(kb)

            # return kb

        else:
            # use the kb defined in args
            with open(f'{self.dataset}_knowledge_base', 'w') as kb_file:
                kb_file.write(self.knowledge_base)
            # return self.knowledge_base

    def __str__(self):
        """ print the stats """
        stats = ''
        stats += f'======================================Statistics for {self.dataset} ====================================\n'
        stats += '====================================== Clause Compliance =========================================\n'

        for key in self.keys:
            for clause in range(self.data.num_classes):
                stats += f'=========== {key} ===========\n'
                stats += f'Clause Compliance of Clause Class_{clause}(x) AND Link(x.y) --> Class_{clause}(y): {self.clause_stats[clause].compliance[key]}' + '\n'

        stats += '\n' + '\n'
        stats += '====================================== Class Distribution  ==========================================\n'
        for key in self.keys:
            for clause in range(self.data.num_classes):
                stats += f'=========== {key} ===========\n'
                stats += f' Class Quantity of Class_{clause}: {self.clause_stats[clause].quantity[key]} \n'
        # print(stats)
        return stats

    def save_data_stats(self):
        """ saves compliance and quantity in a txt file """
        with open(f'{self.dataset}_data_stats', 'w') as file:
            file.write(self.__str__())
