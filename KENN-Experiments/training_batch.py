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

    total_loss = 0
    # total_examples = total_correct = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.relations)[batch.train_mask]
        loss = criterion(out, batch.y.squeeze(1)[batch.train_mask])
        loss.backward()
        optimizer.step()
        model.apply(range_constraint)

        # total_examples += batch.batch_size
        # total_correct += int((out.argmax(dim=-1) == batch.y.squeeze(1)[:batch.batch_size]).sum())
        total_loss += float(loss.item())

    return total_loss / len(train_loader)


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
    # todo maybe make separate loaders for train valid and test in evaluation
    # todo test this method
    # todo progress bar
    # todo make sure that right arguments are returned !

    model.eval()
    epoch_acc = epoch_loss = 0

    preds = []
    for batch in loader:
        # todo keep track of loss
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.relations)[:batch.batch_size]

        y_pred = out.argmax(dim=-1, keepdim=True)
        preds.append(y_pred.cpu())

    preds = torch.cat(preds, dim=0)

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

    # return epoch_acc / len(loader), epoch_loss / len(loader)
    return train_acc, valid_acc, test_acc
