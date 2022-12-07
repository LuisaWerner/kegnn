### Plan to do experiments on KENN

## Research Question number 1: How does KENN perform on other datasets and with other non-relational base models? 
Datasets: all where FB is possible
Base Models to consider: MLP, Logistic Regression
KENN Models: KENN_MLP, KENN_LR

# Hyperparameters to optimize for each dataset 
MLP:        batch size, dropout, hidden channels, lr, num_layers, use node embedding
LR:         don't optimize, don't "train" , just fit once to training data 

for kenn models, take parameters from Base NN as given 
KENN_MLP:   num_neighbors, num_layers_sampling, num_kenn_layers, binary_preactivations
KENN_LR:    num_neighbors, num_layers_sampling, num_kenn_layers, binary_preactivations

## Research Question Number 2: How does KENN work with relational models as base models
Datasets: all where FB is possible
Base Models to consider: GCN, GAT

# Hyperparameters to optimize for each dataset 
Datasets: all where FB is possible
GCN:        batch size, dropout, hidden channels, lr, num_layers, use node embedding, num_neighbors, num_layers_sampling, normalize_edges 
GAT:        batch size, dropout, hidden channels, lr, num_layers, use node embedding, num_neighbors, num_layers_sampling, normalize_edges, num_attention_heads

for kenn models, take parameters from Base NN as given 
KENN_MLP:   num_kenn_layers, binary_preactivations, initialization clause weights? 
KENN_LR:    num_kenn_layers, binary_preactivations, initialization clause weights? 

## Research Question Number 3: How does KENN react if we delete edges
Datasets: all where FB is possible
Models: GCN, GAT, MLP, KENN_MLP, KENN_GCN, KENN_GAT, KENN_LR
# Take models with their previously optimized hyperparameters: 
# try different edges_drop_rates 

## Research Question Number 4: How does KENN deal with different sets of knowledge
Datasets: all where FB is possible
Models: GCN, GAT, MLP, KENN_MLP, KENN_GCN, KENN_GAT, KENN_LR
# Take models with their previously optimized hyperparameters: 
# Experiment with compliance_range, knowledge_filter_key, quantity_range

## Research Question Number 5: How does KENN work on large Graphs
Datasets: One where fullbatch is available for comparison, otherwise a dataset where FB is not available
Models: ClusterGCN, GraphSAINT, GraphSAGE, others with restrictive sampling parameters

# Hyperparameter optimization: 
others/GraphSAGE: same parameters as before, set sampling parameters fixed to make sure it fits in memory
ClusterGCN: additional parameters: cluster_partition_size, num_parts (?)
GraphSAINT: use_norm, sample coverage, walk length 

