To run this script (KENN with MLP as base NN )

dataset = 'ogbn-arxiv' or 'ogbn-products' mode = 'transductive' or 'inductive'

Example:
python train_and_evaluate_KENN.py dataset='ogbn-arxiv' --epochs=500 --runs=10 --mode='transductive' --batch_size=1000
--sampling_neighbor_size=10

To start tensorboard (choose right parameters)

1. go to project directory ./arxiv
2. tensorboard --logdir=runs/ogbn-arxiv/transductive/run0

