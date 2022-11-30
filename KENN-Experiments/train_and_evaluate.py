
from time import time
# import torch.backends.mps
import torch.nn.functional as F
import torch_geometric
from torch.utils.tensorboard.writer import SummaryWriter
import wandb
from app_stats import RunStats, ExperimentStats
from model import get_model
from evaluate import Evaluator
from preprocess_data import *
from training_batch import train, test
from knowledge import *


def callback_early_stopping(valid_accuracies, epoch, args):
    """
    Takes as argument the list with all the validation accuracies.
    If patience=k, checks if the mean of the last k accuracies is higher than the mean of the
    previous k accuracies (i.e. we check that we are not overfitting). If not, stops learning.
    @param valid_accuracies - list(float) , validation accuracy per epoch
    @param epoch: current epoch
    @param args: argument file [Namespace]
    @return bool - if training stops or not
    """
    step = len(valid_accuracies)
    patience = args.es_patience // args.eval_steps
    # no early stopping for 2 * patience epochs
    if epoch < 2 * args.es_patience:
        return False

    # Mean loss for last patience epochs and second-last patience epochs

    mean_previous = np.mean(valid_accuracies[step - 2 * patience:step - patience])
    mean_recent = np.mean(valid_accuracies[step - patience:step])
    delta = mean_recent - mean_previous
    if delta <= args.es_min_delta:
        print("*CB_ES* Validation Accuracy didn't increase in the last %d epochs" % args.es_patience)
        print("*CB_ES* delta:", delta)
        print(f"callback_early_stopping signal received at epoch {epoch}")
        print("Terminating training")
        return True
    else:
        return False


def run_experiment(args):
    torch_geometric.seed_everything(args.seed)
    print(f"backend available {torch.backends.mps.is_available()}")
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Cuda available? {torch.cuda.is_available()}, Number of devices: {torch.cuda.device_count()}')

    print(f'Start {args.mode} Training')
    xp_stats = ExperimentStats()

    test_accuracies = []

    for run in range(args.runs):

        print(f"Run: {run} of {args.runs}")
        writer = SummaryWriter('runs/' + args.dataset + f'/{args.mode}/run{run}')

        model = get_model(args).to(device)
        model.reset_parameters()
        evaluator = Evaluator(name=args.dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = F.nll_loss

        train_losses, valid_losses, train_accuracies, valid_accuracies, epoch_time = [], [], [], [], []

        clause_weights_dict = None # todo needed ?

        if not args.full_batch:
            print(f"Number of Training Batches with batch_size = {args.batch_size}: {len(model.train_loader)}")

        for epoch in range(args.epochs):
            start = time()
            train(model, optimizer, device, criterion)
            end = time()

            if epoch % args.eval_steps == 0:
                _, t_accuracy, v_accuracy, t_loss, v_loss, _ = test(model, criterion, device, evaluator)

                # Save stats for tensorboard
                writer.add_scalar("loss/train", t_loss, epoch)
                writer.add_scalar("loss/train", t_loss, epoch)
                writer.add_scalar("loss/valid", v_loss, epoch)
                writer.add_scalar("accuracy/train", t_accuracy, epoch)
                writer.add_scalar("accuracy/valid", v_accuracy, epoch)

                train_accuracies += [t_accuracy]
                valid_accuracies += [v_accuracy]
                train_losses += [t_loss]
                valid_losses += [v_loss]
                epoch_time += [end - start]

                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {t_loss:.4f}, '
                      f'Time per Train Step: {end - start:.6f} '
                      f'Train: {100 * t_accuracy:.2f}%, '
                      f'Valid: {100 * v_accuracy:.2f}% ')

            # early stopping
            if args.es_enabled and callback_early_stopping(valid_accuracies, epoch, args):
                print(f'Early Stopping at epoch {epoch}.')
                break

        test_accuracy, *_ = test(model, criterion, device, evaluator)
        test_accuracies += [test_accuracy]
        rs = RunStats(run, train_losses, train_accuracies, valid_losses, valid_accuracies, test_accuracy, epoch_time,
                      test_accuracies)
        xp_stats.add_run(rs)
        print(rs)
        wandb.log(rs.to_dict())
        wandb.run.summary["test_accuracies"] = test_accuracies
        writer.close()

    xp_stats.end_experiment()
    print(xp_stats)
    wandb.log(xp_stats.to_dict())


