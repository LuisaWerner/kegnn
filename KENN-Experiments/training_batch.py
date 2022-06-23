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
    total_examples = total_loss = total_correct = 0

    for batch in iter(train_loader):  # todo: do we need iter?
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.relations)[:batch.batch_size]
        loss = criterion(out, batch.y.squeeze(1)[:batch.batch_size])
        loss.backward()
        optimizer.step()
        # model.apply(range_constraint)

        total_examples += batch.batch_size
        total_loss += float(loss)
        total_correct += int((out.argmax(dim=-1) == batch.y.squeeze(1)[:batch.batch_size]).sum())

        print("Outside: input size", batch.size(), "output_size",
              out.size())  # todo this is only to see if both GPUs are used

    epoch_loss = total_loss / len(train_loader)  # corresponds to num_batches

    return epoch_loss


@torch.no_grad()
def test(model, loader, criterion, device, evaluator):
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
    epoch_acc = epoch_loss = 0
    for batch in iter(loader):  # todo: do we need iter?
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.relations)[:batch.batch_size]
        loss = criterion(out, batch.y.squeeze(1)[:batch.batch_size])
        y_pred = out.argmax(dim=-1, keepdim=True)
        batch_acc = evaluator.eval({
            'y_true': batch.y[:batch.batch_size],
            'y_pred': y_pred,
        })['acc']
        epoch_acc += batch_acc
        epoch_loss += loss
        print("Outside: input size", batch.size(), "output_size",
              out.size())  # todo this is only to see if both GPUs are used
    return epoch_acc / len(loader), epoch_loss / len(loader)
