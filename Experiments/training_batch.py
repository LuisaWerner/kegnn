import torch


def train(model, optimizer, device, criterion):
    """
    training loop - trains specified model by computing batches
    returns average epoch accuracy and loss
    parameters are updated with gradient descent
    @param model: a model specified in model.py
    @param optimizer: torch.optimizer object
    @param device: gpu or cpu
    @param criterion: defined loss function
    @param args: input parameters
    """
    model.train()
    total_loss = 0

    for i_batch, batch in enumerate(model.train_loader):

        batch.to(device)
        if batch.train_mask.sum() == 0:
            print('sampled batch does not contain any train nodes')
            continue

        optimizer.zero_grad()

        if 'SAINT' in model.name:

            # if sample_coverage is 0, no normalization coefficients are calculated
            if not model.use_norm or not hasattr(batch, 'edge_norm') or not hasattr(batch, 'node_norm'):
                out = model(batch.x, batch.edge_index, batch.relations).log_softmax(dim=-1)
                loss = criterion(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask])

            # if normalization coefficients are calculated
            else:
                batch.edge_weight = batch.edge_norm * batch.edge_weight
                out = model(batch.x, batch.edge_index, batch.relations, batch.edge_weight).log_softmax(dim=-1)
                loss = criterion(out, batch.y.squeeze(1), reduction='none')
                loss = (loss * batch.node_norm)[batch.train_mask].sum()

            total_loss += float(loss.item())

        else:
            out = model(batch.x, batch.edge_index, batch.relations, batch.edge_weight).log_softmax(dim=-1)
            loss = criterion(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask])
            total_loss += float(loss.item())

        loss.backward()
        optimizer.step()
        print(f'Training: Batch {i_batch} of {len(model.train_loader)} completed')


@torch.no_grad()
def test(model, criterion, device, evaluator):
    """
    validation loop. No gradient updates
    returns accuracy per epoch and loss
    @param model: a model specified in model.py
    @param evaluator: a evaluator instance
    @param data: a data instance
    @param device: gpu or cpu
    @param criterion: defined loss function
    """
    model.eval()
    preds, logits = [], []
    i = 0
    for batch in model.test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.relations, batch.edge_weight).log_softmax(dim=-1)[
              :batch.batch_size]
        logits.append(out.cpu())
        print(f'Evaluating: Batch {i} of {len(model.test_loader)} completed')
        i = i + 1

    all_logits = torch.cat(logits, dim=0)

    train_loss = criterion(all_logits[model.data.train_mask], model.data.y.squeeze(1)[model.data.train_mask]) / len(all_logits)
    valid_loss = criterion(all_logits[model.data.val_mask], model.data.y.squeeze(1)[model.data.val_mask]) / len(all_logits)
    test_loss = criterion(all_logits[model.data.test_mask], model.data.y.squeeze(1)[model.data.test_mask]) / len(all_logits)

    train_acc = evaluator.eval({
        'y_true': model.data.y[model.data.train_mask],
        'y_pred': all_logits.argmax(dim=-1, keepdim=True)[model.data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': model.data.y[model.data.val_mask],
        'y_pred': all_logits.argmax(dim=-1, keepdim=True)[model.data.val_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': model.data.y[model.data.test_mask],
        'y_pred': all_logits.argmax(dim=-1, keepdim=True)[model.data.test_mask]
    })['acc']

    return test_acc, train_acc, valid_acc, train_loss, valid_loss, test_loss

