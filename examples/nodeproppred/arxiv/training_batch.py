import torch
from tqdm import tqdm


def train(model, train_loader, optimizer, device, criterion, args, range_constraint):
    model.train()
    total_examples = total_loss = total_correct = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index')
        out = model(batch.x, batch.adj_t, batch.relations)[
              :args.batch_size]  # the first nodes of the batch are for prediction
        loss = criterion(out, batch.y.squeeze(1)[:args.batch_size])
        loss.backward()
        optimizer.step()
        model.apply(range_constraint)

        total_examples += args.batch_size
        total_loss = + float(loss) * args.batch_size
        total_correct += int((out.argmax(dim=-1) == batch.y.squeeze(1)[:args.batch_size]).sum())
        # total_correct += int(out.argmax(dim=-1) == batch.y[:args.batch_size])

    epoch_acc = total_correct / total_examples
    epoch_loss = total_loss / total_examples

    # print(get_stats_summary())
    return epoch_acc, epoch_loss


@torch.no_grad()
def test(model, test_loader, criterion, args, device):
    """
    returns (accuracy, loss)
    """
    model.eval()

    total_examples = total_loss = total_correct = 0
    for batch in test_loader:
        batch = batch.to(device, 'edge_index')
        out = model(batch.x, batch.adj_t, batch.relations)[:args.batch_size]
        loss = criterion(out, batch.y.squeeze(1)[:args.batch_size])

        total_examples += args.batch_size
        total_correct += int((out.argmax(dim=-1) == batch.y.squeeze(1)[:args.batch_size]).sum())
        total_loss = + float(loss) * args.batch_size

    epoch_acc = total_correct / total_examples
    epoch_loss = total_loss / total_examples

    return epoch_acc, epoch_loss
