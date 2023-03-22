## Knowledge Enhanced Graph Neural Networks 
Graph data is omnipresent and represents a large variety of use cases. 
Though rich in information, graphs are often noisy, incomplete and complex. 
Therefore, graph learning tasks such as node classification or graph completion have gained increasing attention in the recent past. 
On the one hand, neural methods such as graph neural networks have proven to be a robust method for learning rich representations of noisy graphs. 
On the other hand, symbolic methods enable exact reasoning on graphs. 
We propose KeGNN, a neuro-symbolic framework for learning on graph data that combines both paradigms and allows for the integration of prior knowledge into a graph neural network model. 
In essence, KeGNN consists of a graph neural network as a base on which knowledge enhancement layers are stacked with the objective of refining the predictions with respect to prior knowledge. 
We instantiate KeGNN in conjunction with two standard graph neural networks: Graph Convolutional Networks and Graph Attention Networks, and apply KeGNN to multiple benchmark datasets for node classification. 

KeGNN is an extension of KENN [Knowledge Enhanced Neural Networks (KENN)](https://github.com/rmazzier/KENN-Citeseer-Experiments) on the [Citeseer Dataset](https://linqs.soe.ucsc.edu/data).
It corresponds to the experiments in KENN by setting --dataset to 'CiteSeer' and --model to 'KENN_MLP'. 

This implementation is based on [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric). 

## How to run the code 
### Environment Setup
In order to make sure that the right environment is used, the necessary Python packages and their versions are specified in `requirements.txt`. We use Python 3.9. To install them go in the project directory and create a conda environment with the following packages. 
```
pip install -r requirements.txt
``` 

### Run the Experiments
We use [Weights and Biases](https://wandb.ai/site) (WandB) as experiment tracking tool. The experiments can be run WITHOUT or WITH  the use of WandB.
For internal use, I suggest running WITH WandB to keep track of results. WandB can be temporarely disabled by setting `mode='disabled'` in `wandb.init(...)` in `run_experiments.py`.
If you want to use weights and biases specify the following parameters in  `conf.json`.
```
"wandb_use" : true,
"wandb_label": "<your label>",
"wandb_project" : "<your project>",
"wandb_entity": "<your entity>"
```

Then use the following command to run the experiments: 
```
cd Experiments
python run_experiments.py conf.json
```

The experiments can be interpreted directly on weights and biases or can be further analyzed with the jupyter notebook `results_analysis.ipynb` in the `notebook` directory. 

For running the code without weights and biases do the following steps: 
```
cd Experiments
python train_and_evaluate.py conf.json 
```



## How to adapt the conf file
The file conf.json 
-- dataset
choose from ['CiteSeer', 'Cora', 'PubMed','ogbn-products', 'ogbn-arxiv', 'Reddit2', 'Flickr', 'AmazonProducts, 'Yelp]
Note: CiteSeer, Cora, PubMed and Flickr are tested with this implementation

| Name           | Description                      | #nodes    | #edges      | #features | #Classes | Task                 |
|----------------|----------------------------------|-----------|-------------|-----------|----------|----------------------|
| CiteSeer       | from Planetoid, Citation Network | 3,327     | 9,104       | 3,703     | 6        | Node classification  |
| Cora           | from Planetoid, Citation Network | 2,708     | 10,556      | 1,433     | 7        | Node Classification  |
| PubMed         | from Planetoid, Citation Network | 19,717    | 88,648      | 500       | 3        | Node Classification  |
| ogbn-products  | from OGB, co-purchase network    | 2,449,029 | 61,859,140  |           | 1        | Node Classification  |
| ogbn-arxiv     | from OGBm, Citation Network      | 169,343   | 1,166,243   |           |          | Node Classification  |
| Reddit2        | from GraphSAINT                  | 232,965   | 23,213,838  | 602       | 41       | Node Classification  |
| Flickr         | from GraphSAINT                  | 89,250    | 899,756     | 500       | 7        | Node Classification  |
| AmazonProducts | from GraphSAINT                  | 1,569,960 | 264,339,468 | 200       | 107      | Node Classification  |
| Yelp           | from GraphSAINT                  | 716,847   | 13,954,819  | 300       | 100      | Node Classification  |

For datasets coming from Planetoid, the splitting procedure can be chosen with --planetoid_split (see PyG documentation)
--use_node_embedding: What is this needed for? 
--normalize_edges: if True, noramlize the loaded edges, default: False

### Knowledge Generation
The clauses are created following the schema $\forall x,y: Class(x) \land Cite(x,y) \implies Class(y)$. 
The parameters --compliance_range, --quantity_range and --knowledge_filter_key allow to generate the clauses depending on the statistics of the underlying datasets. 
By default, for each class a clause is created. To include all clauses, the ranges are set to $[0.0, 1.0]$.
By adapting the ranges, only clauses are created that have a clause compliance or relative representation in the dataset. 
The knowledge filter key determines on which subset the statistics should be calculated. It can be set to "all", "train", "val" or "test". 
Be aware of that taking samples from the test set might lead to overoptimistic results.
If you want to use your custom knowledge_base, you can set the parameter --knowledge_base to the string that describes the knowledge you want to use

### Randomly Dropping Edges during training 
It might be interested to see, if KE layers need to improvements when training data is scarce. In this case it might be desireable to randomly drop edges in the training set. 
The parameter --edges_drop__rate can be chosen in [0.0, 1.0] to remove eges from the train set. 

### Training Parameters
--lr: learning rate, default: 0.01
--dropout: dropout rate, default 0.5
--epochs: number of epochs per iteration
--iterations: number of iterations. At each iterations, parameters are re-initialized and training set is preprocessed again
--eval_steps: How often should model be evaluated, defaut=10

#### Early Stopping: Stop training if validation accuracy doesn't improve considerably
--es_enabled: use early stopping or not, default: true
--es_min_delta: threshold for improvement, default: 0.001
--es_patience: epochs to wait until early stopping is activated, default: 10

--mode: "inductive" or "transductive", default "transductive"

#### Batch Training
--full-batch: set to true to train in full-batch mode. Might throw memory exceptions, default=false
--batch_size: size of the batch in batch training, default: batch_size

### Model Parametres
#### test_loader: 
--num_workers, default: 0
--num_neighbors: a list of number of neighbors to sample per layer 
--num_layers_sampling: 1

#### Model architecture: 
--num_layers: Number of all layers (todo only hidden layers)
--hidden_channels: input for hidden layers, default=256f

### Linear Regression
train_loader: Standard Neighbor Loader (like test Loader)

### Logistic Regression 
train_loader: Standard Neighbor Loader (like test Loader)

### MLP
train_loader: Standard Neighbor Loader (like test Loader) 

### Standard (Original MLP setting of Relational KENN paper)
train_loader: Standard Neighbor Loader (like test Loader)

### GCN (Graph Convolutional Network)
train_loader: Standard Neighbor Loader (like test Loader)

### GraphSAGE 
train_loader: Standard Neighbor Loader (like test Loader)
### GAT (Graph Attention Network)
train_loader: Standard Neighbor Loader (like test Loader)
--attention_heads: Number of in_heads for attention, default: 8

### ClusterGCN
train_loader: ClusterLoader
--num_parts: number of partitions, default: 100

### GraphSAINT
train_loader: Random Walk Sampler 
--sample_coverage: on how many nodes calculate normalization statistics
if zero, no coefficients are calculated , default: 10 
--num_steps: nunber of random walk steps, default: 5
--walk_length: default: 3
--use_norm: if True, set 'add' as aggregator in convolution, else 'mean'

### KENN Models
--binary_preactivation: value for binary preactivation 
--initial_clause_weight: todo 
--num_kenn_layers, default = 30
--range_constrain_lower / --range_constraint_upper: values to clip the clause weight, default: 0, 500

### Other parameters
--seed: random seed, default 100
--save_results:  store_true if set
--use_node_embedding: store_true if set
--save_data_stats: store_true if set 











 








