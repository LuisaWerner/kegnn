import torch


def train(model, train_loader, optimizer, device, criterion, range_constraint):
    """
    training loop - trains specified model by computing batches
    returns average epoch accuracy and loss
    parameters are updated with gradient descent

    @param model: a model specified in model.py
    @param train_loader: A PyG NeighborLoader object that returns batches. The first [batch_size] nodes in the batch are target nodes.
    The following ones are the sampled Neighbors
    @param optimizer: torch.optimizer object
    @param device: gpu or cpu
    @param criterion: defined loss function
    @param args: input parameters
    @param range_constraint : weight clipping for parameters
    """
    model.train()

    for batch in train_loader:
        batch = batch.to(device)  # todo reduce relations attribute
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index,
                    batch.relations)  # the target nodes have always to be the first |batch_size| nodes
        loss = criterion(out[:batch.batch_size], batch.y.squeeze(1)[:batch.batch_size])
        loss.backward()
        optimizer.step()
        model.apply(range_constraint)


@torch.no_grad()
def test(model, loader, criterion, device, evaluator, data):
    """
    validation loop. No gradient updates
    returns accuracy per epoch and loss

    @param model: a model specified in model.py
    @param loader: A PyG NeighborLoader object that returns batches.
    The first [batch_size] nodes in the batch are target nodes.
    The following ones are the sampled Neighbors
    @param device: gpu or cpu
    @param criterion: defined loss function
    """
    model.eval()
    preds, logits = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.relations)[:batch.batch_size]
        logits.append(out.cpu())

    all_logits = torch.cat(logits, dim=0)
    preds = all_logits.argmax(dim=-1, keepdim=True)

    train_loss = criterion(all_logits[data.train_mask], data.y.squeeze(1)[data.train_mask])
    valid_loss = criterion(all_logits[data.valid_mask], data.y.squeeze(1)[data.valid_mask])
    test_loss = criterion(all_logits[data.test_mask], data.y.squeeze(1)[data.test_mask])

    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': preds[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.valid_mask],
        'y_pred': preds[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': preds[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss
