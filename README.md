# Knowledge Enhanced Graph Neural Networks

This repository contains the implementation of [Knowledge Enhanced Graph Neural Networks (KeGNN)] (https://openreview.net/pdf?id=7cdXVj9z6Y) and the experiments. 
This is a work by the [Tyrex Team](https://tyrex.inria.fr/). It is as an accepted paper at the [KBCG Workshop](https://knowledgeai.github.io/) at IJCAI'23. 

Graph data is omnipresent and has a large variety of applications such as natural science, social networks or semantic web. 
Though rich in information, graphs are often noisy and incomplete. Therefore, graph completion tasks such as node classification or link prediction have gained attention. 
On the one hand, neural methods such as graph neural networks have proven to be robust tools for learning rich representations of noisy graphs. 
On the other hand, symbolic methods enable exact reasoning on graphs. 
We propose KeGNN, a neuro-symbolic framework for learning on graph data that combines both paradigms and allows for the integration of prior knowledge into a graph neural network model. 
In essence, KeGNN consists of a graph neural network as a base on which knowledge enhancement layers are stacked with the objective of refining predictions with respect to prior knowledge. 
We instantiate KeGNN in conjunction with two standard graph neural networks: Graph Convolutional Networks and Graph Attention Networks, and evaluate KeGNN on multiple benchmark datasets for node classification.

We apply KeGNN to the following datasets that are benchmarks for node classification.
The datasets are publicly available at the dataset collection of [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric). 

| Name     | Description                        | #nodes | #edges  | #features | #Classes | Task                 |
|----------|------------------------------------|--------|---------|-----------|----------|----------------------|
| CiteSeer | from Planetoid, Citation Network   | 3,327  | 9,104   | 3,703     | 6        | Node classification  |
| Cora     | from Planetoid, Citation Network   | 2,708  | 10,556  | 1,433     | 7        | Node Classification  |
| PubMed   | from Planetoid, Citation Network   | 19,717 | 88,648  | 500       | 3        | Node Classification  |
| Flickr   | from GraphSaint [1], Image Network | 89,250 | 899.756 | 500       | 7        | Node Classification  |

### Before running the experiments in both implementations
1. In order to make sure that the right environment is used, the necessary Python packages and their versions are specified in `requirements.txt`. 
We use Python 3.9. 
To install them go in the project directory and create a conda environment with the following packages. 
```
pip install -r requirements.txt
``` 

### To run the Experiments 
We use [Weights and Biases](https://wandb.ai/site) (WandB) as experiment tracking tool. The experiments can be run WITHOUT or WITH  the use of WandB.
## A. Without Wandb
1. To run the experiments without WandB, run the following command. 

```
cd Experiments
python train_and_evaluate.py conf.json 
```

(By default, ```"wandb_use" : false``` is set in `re-implementation/conf.json`)  

The results can be viewed and visualizes with a Jupyter notebook.
The model and the dataset name have to be set manually in the first cells of the notebook.
The notebook can be found in 
```
cd Experiments/notebooks
inspect_results_.ipynb
```


## B. With Wandb
2. If you want to use weights and biases specify the following parameters in  `Experiments/conf.json`.
```
"wandb_use" : True,
"wandb_label": "<your label>",
"wandb_project" : "<your project>",
"wandb_entity": "<your entity>"
```

Then use the following command to run the experiments: 
```
cd Experiments
python run_experiments.py conf.json
```

The results can be viewed and visualizes with a Jupyter notebook.
Remark that the url to the weights and biases database need to be adapted according to your `wandb project` and `wandb entity` and the correct run id needs to be set. 
Currently, we put this information for our wandb repository. 
The notebook can be found in 
```
cd Experiments/notebooks
inspect_results_wandb.ipynb
```

## How to configure the experiments? 
The settings and parameters of the run can be specified in a configuration file `Experiments/conf.json` as follows.

### Experiment setup 
- dataset: The dataset on which the experiments are conducted. Values ['CiteSeer', 'Cora', 'PubMed', 'Flickr'] - Default: 'CiteSeer'
- device: GPU number in case of available GPUS. Values: positive integers - Default: 0
- model: Model for the experiments. Values: ['KeGCN', 'KeGAT', 'KeMLP', 'GCN', 'GAT', 'MLP'] - Default: 'KeGCN'
- eval_steps. How often to evaluate and calculate validation and test accuracy (Each x-th epoch). Values: positive integers - Default: 1 
- planetoid split: Split indices into train/valid/test for the Plantoid datasets. See [here](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid) for details. Default: 'full'
- runs: Number of independent runs to conduct. Values: positive integers. Default: 50
- seed: Random seed. Values: positive integers. Default: 1234

### Standard Hyperparameters
- adam_beta1: Adam Optimizer Parameter Beta1. Default: 0.9
- adam_beta2: Adam Optimizer Parameter Beta2. Default: 0.99
- adam_eps: Adam Optimizer Parameter Epsilon. Default: 1e-07
- attention_heads: Number of attention heads for multi-head attention. Values: positive integers. Default: 8
- batch_size: Batch Size for Mini-Batch Training. Small Batches increase runtimes. Only used when full_batch is set to False. Positive Integers. Default: 512
- dropout: Dropout rate to avoid overfitting. Real numbers in [0,0.9] recommended. Default: 0.5
- edges_drop_rate: Random dropout of edges before training. Real numbers in [0.0, 0.9] recommended. Default: 0.0
- epochs: Number of training epochs. Positive integers. Default: 200
- es_enabled: Early stopping mechanism. Values: [true, false]. Default: true
- es_min_delta: Early stopping minimum delta between validation accuracy of previous steps and current validation accuracy. Small positivereal valued numbers. Default: 0.001.
- es_patience: number of epochs before early stopping can be activated. Positive integers. Default: 10
- full_batch: activation of full batch training: Values: [true, false]. Default: true
- hidden_channels: Number of neurons in hidden layers of base NN. Values: positive Integers. Default: 128
- lr: learning rate. Real Valued Numbers in [0.0001, 0.1]. Default: 0.01
- normalize_edges: Normalize edges with Degree Matrix. Values: [true, false]. Default: false
- num_layers: Number of hidden layers in base NN. Values: positive Integers. Default: 3


### Knowledge Enhancement specific parameters
- binary_preactivation: Initialization of binary predicate groundings. Values: high positive real-valued numbers. Default: 500.0
- boost function: value: "GodelBoostConormApprox"
- clause weight: Initialization of the clause weight. Values: positive real-valued numbers. Default: 0.5
- min weight: minimum weight for clause weight clipping. Default: 0.0
- max weight: maximum weight for clause weight clipping. Default: 500.0
- num_kenn_layers: number of stacked kenn layers. Values: positive integers. Default: 3


### WandB specific hyperparameters
- wandb_use: flag to use wandb or not. values: [true, false]. Default: false
- wandb_label: runs can be labelled. Put a label value here if you want to label your runs. Values: 'string'. Default: 'test'
- wandb_project: your wandb project
- wandb_entity: your wandb project 








