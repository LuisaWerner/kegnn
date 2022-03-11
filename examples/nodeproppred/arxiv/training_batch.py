import time

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

    start = time.time()
    for batch in iter(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.relations)[:batch.batch_size]
        loss = criterion(out, batch.y.squeeze(1)[:batch.batch_size])
        loss.backward()
        optimizer.step()
        model.apply(range_constraint)

        total_examples += batch.batch_size
        total_loss += float(loss) * batch.batch_size
        total_correct += int((out.argmax(dim=-1) == batch.y.squeeze(1)[:batch.batch_size]).sum())

    epoch_acc = total_correct / total_examples
    epoch_loss = total_loss / total_examples
    end = time.time()
    print(f'epoch time: {end - start}')

    return epoch_acc, epoch_loss


@torch.no_grad()
def test(model, test_loader, criterion, device):
    """
    validation loop. No gradient updates
    returns accuracy per epoch and loss

    @param model: a model specified in model.py
    @param test_loader: A PyG NeighborLoader object that returns batches.
    The first [batch_size] nodes in the batch are target nodes.
    The following ones are the sampled Neighbors
    @param device: gpu or cpu
    @param criterion: defined loss function
    """
    model.eval()

    total_examples = total_loss = total_correct = 0
    for batch in iter(test_loader):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.relations)[:batch.batch_size]
        loss = criterion(out, batch.y.squeeze(1)[:batch.batch_size])

        total_examples += batch.batch_size
        total_correct += int((out.argmax(dim=-1) == batch.y.squeeze(1)[:batch.batch_size]).sum())
        total_loss += float(loss) * batch.batch_size

    epoch_acc = total_correct / total_examples
    epoch_loss = total_loss / total_examples

    return epoch_acc, epoch_loss
