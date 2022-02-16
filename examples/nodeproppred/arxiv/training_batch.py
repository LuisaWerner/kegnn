import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, train_loader, optimizer, device, args, range_constraint):
    model.train()
    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index')
        data.x, data.adj_train, data.relations_train
        out = model(batch.x, batch.edge_index)[:args.batch_size]
        loss = F.cross_entropy(out, batch['paper'].y[:args.batch_size])
        loss.backward()
        optimizer.step()
        model.apply(range_constraint)

        total_examples += args.batch_size
        total_loss = + float(loss) * args.batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(model, test_loader, args, device):
    """
    returns (accuracy, loss)
    """
    model.eval()

    total_examples = total_loss = total_correct = 0
    for batch in tqdm(test_loader):
        batch = batch.to(device, 'edge_index')
        out = model(batch.x_dict, batch.edge_index_dict)['paper'][:args.batch_size]
        loss = F.cross_entropy(out, batch['paper'].y[:args.batch_size])
        pred = out.argmax(dim=-1)

        total_examples += args.batch_size
        total_correct += int((pred == batch['paper'].y[:args.batch_size]).sum())
        total_loss = + float(loss) * args.batch_size

    return total_correct / total_examples, total_loss / total_examples
