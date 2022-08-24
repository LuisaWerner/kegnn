import torch


def train(model, train_loader, optimizer, device, criterion, args):
    # todo: can we take edge weights as binary preactivations for kenn-sub if we have them ?
    # todo: or multiply binary preactiations by edge weights
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
    total_loss = total_examples = 0

    for batch in train_loader:

        optimizer.zero_grad()

        # batch = batch.to(device)
        if batch.train_mask.sum() == 0:
            print('sampled batch does not contain any train nodes')
            continue

        if args.train_sampling == 'cluster':
            out = model(batch.x, batch.edge_index, batch.relations, None)  # none for edge weight ?
            loss = criterion(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask])
            total_loss += loss.item() * torch.sum(batch.train_mask)
            total_examples += torch.sum(batch.train_mask).item()  # some nodes might be sampled more than once

        elif args.train_sampling == 'graph_saint':

            # if sample_coverage is 0, no normalization coefficients are calculated
            if not hasattr(batch, 'edge_norm'):
                out = model(batch.x, batch.edge_index, batch.relations)
                loss = criterion(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask])

            # if normalization coefficients are calculated
            else:
                if batch.edge_weight is None:
                    batch.edge_weight = torch.ones(batch.edge_index.size()[1])

                batch.edge_weight = batch.edge_norm * batch.edge_weight  # todo how does this affect kenn-sub
                out = model(batch.x, batch.edge_index, batch.relations, batch.edge_weight)
                loss = criterion(out, batch.y.squeeze(1), reduction='none')
                loss = (loss * batch.node_norm)[batch.train_mask].sum()

            total_loss += loss.item() * batch.num_nodes
            total_examples += batch.num_nodes

        else:
            # The batches are created with Neighbor Loader
            # the target nodes have always to be the first |batch_size| nodes
            # each node is only taken into account as target nodes once, while it can be neighbor several times
            # we are not able to just select by train_mask because neighbors would contribute to loss more than ONCE
            batch.x.to(device)
            batch.edge_index.to(device)
            batch.relations.to(device)
            out = model(batch.x, batch.edge_index, batch.relations)
            loss = criterion(out[:batch.batch_size], batch.y.squeeze(1)[:batch.batch_size])
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size

        loss.backward()
        optimizer.step()
        # model.apply(range_constraint)

    return total_loss / total_examples


@torch.no_grad()
def test(model, loader, criterion, device, evaluator, data):
    # TODO: why is in graph_saint.py and cluster_gcn.py iterated over convolutions?
    # todo: maybe a property of the Neighbor Sampler?
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
    # preds = all_logits.argmax(dim=-1, keepdim=True)[data.train_mask]

    train_loss = criterion(all_logits[data.train_mask], data.y.squeeze(1)[data.train_mask])
    valid_loss = criterion(all_logits[data.val_mask], data.y.squeeze(1)[data.val_mask])
    test_loss = criterion(all_logits[data.test_mask], data.y.squeeze(1)[data.test_mask])

    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': all_logits.argmax(dim=-1, keepdim=True)[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': all_logits.argmax(dim=-1, keepdim=True)[data.val_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': all_logits.argmax(dim=-1, keepdim=True)[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss
