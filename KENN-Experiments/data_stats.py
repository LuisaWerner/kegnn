import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def get_y_stats(data: Data, key: str, n_rows=40):
    """
    Returns an overview of the class distribution in the data set as string
    @param data: PyG data object
    @param key: 'all', 'train', 'val', 'test'
    @param n_rows: how many rows to print/save
    """
    if key == 'all':
        y = data.y.cpu().detach().numpy()
    else:
        mask = key + '_mask'
        y = data.y[getattr(data, mask)].cpu().detach().numpy()

    unique, counts = np.unique(y, return_counts=True)
    db = pd.DataFrame.from_dict(dict(zip(unique, counts)), orient='index').sort_values(by=0, ascending=False).rename(
        columns={0: 'count'})

    if db.shape[0] < data.num_classes:
        print('some classes have zero nodes')
    return f'Display the fist {n_rows} rows: \n {db.head(n_rows)}'


def clause_compliance(data: Data, cls: int):
    """ Calculates the clause_compliance
    Clause compliance = # neighbors that both have the respective class cls / # total num neighbors
    @param data: PyG Data Object
    @param cls: class [here defined as integer]
    """
    y = data.y.cpu().detach().numpy()
    edge_index = np.transpose(data.edge_index.cpu().detach().numpy())

    cls_mask = np.where(y == cls)[0]  # the indexes with cls
    cls_list = []
    for node in cls_mask:
        edge_mask = edge_index[np.where(np.logical_or([edge_index[:, 0] == node][0], [edge_index[:, 1] == node][0])), :]
        cls_list.append(np.take(y, edge_mask).squeeze(axis=0))

    cls_list = np.concatenate(cls_list, axis=0)
    n_neighbors = len(cls_list)
    n_neighbors_equal = len(np.where(cls_list[:, 0] == cls_list[:, 1])[0])

    return n_neighbors_equal / n_neighbors


def save_data_stats(data: Data, args):
    """
    writes and saves the stats in string file
    @param data: PyG data object
    @param args: Input parameters
    """
    stats = ''
    stats += f'======================================Statistics for {args.dataset} ====================================' + '\n'
    stats += '====================================== Clause Compliance =========================================' + '\n'
    stats += '=========== Overall ===========' + '\n'
    for i in range(data.num_classes):
        stats += f'Clause Compliance of Clause Class_{i}(x) AND Cite(x.y) --> Class_{i}(y): {clause_compliance(data, i)}' + '\n'

    stats += '\n' + '=========== Train ===========' + '\n'
    data_train = Data(x=data.x[data.train_mask], y=data.y[data.train_mask],
                      edge_index=subgraph(data.train_mask, data.edge_index, relabel_nodes=True)[0])
    for i in range(data.num_classes):
        stats += f'Clause Compliance of Clause Class_{i}(x) AND Cite(x.y) --> Class_{i}(y): {clause_compliance(data_train, i)}' + '\n'

    stats += '\n' + '=========== Valid ===========' + '\n'
    data_val = Data(x=data.x[data.val_mask], y=data.y[data.val_mask],
                    edge_index=subgraph(data.val_mask, data.edge_index, relabel_nodes=True)[0])
    for i in range(data.num_classes):
        stats += f'Clause Compliance of Clause Class_{i}(x) AND Cite(x.y) --> Class_{i}(y): {clause_compliance(data_val, i)}' + '\n'

    stats += '\n' + '=========== Test ===========' + '\n'
    data_test = Data(x=data.x[data.test_mask], y=data.y[data.test_mask],
                     edge_index=subgraph(data.test_mask, data.edge_index, relabel_nodes=True)[0])
    for i in range(data.num_classes):
        stats += f'Clause Compliance of Clause Class_{i}(x) AND Cite(x.y) --> Class_{i}(y): {clause_compliance(data_test, i)}' + '\n'

    stats += '\n' + '\n'
    stats += '====================================== Class Distribution  ==========================================' + '\n'
    stats += '=========== Overall ===========' + '\n'
    stats += get_y_stats(data, 'all') + '\n'
    stats += '=========== Train ===========' + '\n'
    stats += get_y_stats(data, 'train') + '\n'
    stats += '=========== Valid ===========' + '\n'
    stats += get_y_stats(data, 'val') + '\n'
    stats += '=========== Test ===========' + '\n'
    stats += get_y_stats(data, 'test')

    with open('data_stats', 'w') as file:
        file.write(stats)

    print('Done')
