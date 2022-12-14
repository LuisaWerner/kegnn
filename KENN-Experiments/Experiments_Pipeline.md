### Plan to do experiments on KENN
## Base Neural Networks 
| **Dataset/Base NN** | **MLP**                              | **GCN**                                    | **GAT**                           | **ClusterGCN**                        | **GraphSAGE**                         | **GraphSAINT**                        |
|--------------------|--------------------------------------|--------------------------------------------|-----------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|
| **Citeseer**       | Results  Parameters  from KENN paper | Results Parameters from GCN paper          | Results Parameters from GAT paper | X                                     | X                                     | X                                     |
| **Cora**           | TODO  Parameters + Results           | Results Parameters from GCN paper          | Results Parameters from GAT pape  | X                                     | X                                     | X                                     |
| **PubMed**         | TODO  Parameters + Results           | Results Parameters from GCN paper          | Results Parameters from GAT pape  | X                                     | X                                     | X                                     |
| **Flickr**         | TODO Parameters + Results            | Results  parameters from  GAT/Saint paper  | Results Parameters from GAT pape  | X                                     | X                                     | X                                     |
| **ogbn-arxiv**     | Results Parameters from ogb paper    | Results Parameters from ogb paper          | Parameters from GAT paper         | X                                     | X                                     | X                                     |
| **Reddit2**        | X  (OOM on FB?)                      | X  (OOM on FB?)                            | X  (OOM on FB?)                   | Results + Parameters from SAINT paper | Results + Parameters from SAINT paper | Results + Parameters from SAINT paper |
| **Yelp**           | X  (OOM on FB?)                      | X  (OOM on FB?)                            | X  (OOM on FB?)                   | Results + Parameters from SAINT paper | Results + Parameters from SAINT paper | Results + Parameters from SAINT paper |
| **AmazonProducts** | X  (OOM on FB?)                      | X  (OOM on FB?)                            | X  (OOM on FB?)                   | Results + Parameters from SAINT paper | Results + Parameters from SAINT paper | Results + Parameters from SAINT paper |
| **ogbnProducts**   | X  (OOM on FB?)                      | X  (OOM on FB?)                            | X  (OOM on FB?)                   | Results + Parameters from ogbn test   | Results + Parameters from ogbn test   | Results + Parameters from ogbn test   |



## KENN Models
| **Dataset/Base NN** | **KENN_MLP**                         | **KENN_GCN**                 | **KENN_GAT**               | **KENN_ClusterGCN**        | **GraphSAGE**              | **GraphSAINT**             |
|--------------------|--------------------------------------|------------------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| **Citeseer**       | Results  Parameters  from KENN paper | TODO  Parameters + Results   | TODO  Parameters + Results | X                          | X                          | X                          |
| **Cora**           | TODO  Parameters + Results           | TODO  Parameters + Results   | TODO  Parameters + Results | X                          | X                          | X                          |
| **PubMed**         | TODO Parameters Results              | TODO  Parameters + Results   | TODO  Parameters + Results | X                          | X                          | X                          |
| **Flickr**         | TODO  Parameters + Results           | TODO  Parameters + Results   | TODO  Parameters + Results | X                          | X                          | X                          |
| **ogbn-arxiv**     | TODO  Parameters + Results           | TODO  Parameters + Results   | TODO  Parameters + Results | X                          | X                          | X                          |
| **Reddit2**        | X  (OOM on FB?)                      | X  (OOM on FB?)              | X  (OOM on FB?)            | TODO  Parameters + Results | TODO  Parameters + Results | TODO  Parameters + Results |
| **Yelp**           | X  (OOM on FB?)                      | X  (OOM on FB?)              | X  (OOM on FB?)            | TODO  Parameters + Results | TODO  Parameters + Results | TODO  Parameters + Results |
| **AmazonProducts** | X  (OOM on FB?)                      | X  (OOM on FB?)              | X  (OOM on FB?)            | TODO  Parameters + Results | TODO  Parameters + Results | TODO  Parameters + Results |
| **ogbnProducts**   | X  (OOM on FB?)                      | X  (OOM on FB?)              | X  (OOM on FB?)            | TODO  Parameters + Results | TODO  Parameters + Results | TODO  Parameters + Results |

# 0. Which datasets are considered as large-scale or not?
Test full-batch GCN with all neighbors on one iteration, epoch for OOM
Analyze how GPU memory is used 

# 1. Use Hyperparameter for base NN already applied before for comparability
- Use parameters that were proposed in previously published approaches
GCN paper: see here https://github.com/tkipf/gcn/blob/master/gcn/train.py
GAT paper: see here https://github.com/PetarV-/GAT/blob/master/execute_cora.py
ogbn paper: see here https://github.com/pyg-team/pytorch_geometric
GraphSAINT paper: see here https://github.com/GraphSAINT/GraphSAINT + Paper
KENN paper : see here https://github.com/DanieleAlessandro/KENN
Use the hyperparameters for the base NN also for KENN lateron 
- TODO : do we reproduce the results or just take them as they are? 


# 2. Conduct Hyperparameter optimization for KENN specific parameters
These are: 
- initialization of binary preactivations (fixed number, random high number, ...)
- initialization of clause weights (fixed constant number, random number...)
- number of KENN layers 
- choice of boost function (in kenn boost.py)


## Research Question Number 3: How does KENN react if we delete edges
Datasets: all where FB is possible
Models: GCN, GAT, MLP, KENN_MLP, KENN_GCN, KENN_GAT, KENN_LR
# Take models with their previously optimized hyperparameters: 
# try different edges_drop_rates and compare results 


## Research Question Number 4: How does KENN deal with different sets of knowledge
Datasets: all where FB is possible
Models: GCN, GAT, MLP, KENN_MLP, KENN_GCN, KENN_GAT, KENN_LR
# Take models with their previously optimized hyperparameters: 
# Experiment with compliance_range, knowledge_filter_key, quantity_range
Maybe only on scalable models to keep it simple ? Or datasets with not so many classes? 

## Research Question Number 5: Why does KENN not work well in conjunction with GNN models? 
# (or other models depending on obtained results)
Debug a model: (maybe pick only one dataset with not too many clauses etc. ) 
train base NN: analyze which predictions are wrong or right
See what happens to the predictions in the KENN layer
train greedy: separate training of KENN layers 

## Finally: Implementation TODOS 
1. Test OOM for full-batch approaches on large datasets to categorize datasets into large and small graphs
2. Small HP optimization/tests for MLP (don't spend too much time on it)
3. Hyperparameter optimization for KENN models (only optimize KENN-specific parameters and take parameters for Base NN as given)
4. Implement DataParallel/Distributed Parallel
5. Possibility to accelerate KENN layers?
6. Influenced-based mini-batching paper 

