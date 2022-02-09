import torch
import torch.nn.functional as F


def train_inductive(model, data, train_idx, optimizer):
    """
    train_inductive
    training step for transductive setting
    @param model: callable NN model of torch.nn.Module
    @param data: data object with x, y, adjacency matrix (separate matrices for train/valid/test)
    @param optimizer: torch.optim object
    returns: loss (float)
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_train, data.relations)[train_idx]  # take only the train indices
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_inductive(model, data, split_idx, evaluator):
    # todo: adapt for KENN
    """
    test_inductive
    @param model - should be a NN of type torch.nn.module
    @param data - a PyG data object with x, y, adjacency matrix (for train, valid, test)
    @param split_idx - dictionary for split into train/valid/test
    @param evaluator - an evaluator object
    return: accuracy (float) on train, valid, test set
    """
    model.eval()
    out_train = model(data.x, data.adj_train)[split_idx['train']]
    out_valid = model(data.x, data.adj_valid)[split_idx['valid']]
    out_test = model(data.x, data.adj_test)[split_idx['test']]

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': out_train.argmax(dim=-1, keepdim=True),
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': out_valid.argmax(dim=-1, keepdim=True),
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': out_test.argmax(dim=-1, keepdim=True),
    })['acc']

    return [train_acc, valid_acc, test_acc], [out_train, out_valid, out_test]


def train_transductive(model, data, train_idx, optimizer):
    # todo: adapt for KENN
    """
    train_transductive
    training step for transductive setting
    @param model: callable NN model of torch.nn.Module
    @param data: data object with x, y, adjacency matrix (full graph)
    @param optimizer: torch.optim object
    returns: loss (float)
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t, data.relations)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_transductive(model, data, split_idx, evaluator):
    """
    test_transductive
    @param model - should be a NN of type torch.nn.module
    @param data - a PyG data object with x, y, adjacency matrix (all links in full graph)
    @param split_idx - dictionary for split into train/valid/test
    @param evaluator - an evaluator object
    return: accuracy (float) on train, valid, test set
    """
    model.eval()

    out = model(data.x, data.adj_t, data.relations)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc, out
