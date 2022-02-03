import torch
import torch.nn.functional as F


def train_inductive(model, data, train_idx, optimizer):
    """ TODO: Test/Debug and employ in training loop """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_train)[train_idx]  # take only the train indices
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_inductive(model, data, split_idx, evaluator):
    """ TODO - NOT DONE YET """
    model.eval()

    out = model(data.x, data.adj_t)  # todo: replace by data.adj_t
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

    return train_acc, valid_acc, test_acc


def train_transductive(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_transductive(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)  # todo: replace by data.adj_t
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

    return train_acc, valid_acc, test_acc
