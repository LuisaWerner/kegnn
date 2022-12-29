import wandb
from train_and_evaluate import run_experiment


sweep_config = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'valid_acc'
    },
    'parameters': {
        'epochs': {'values': [200]},
        'lr': {'value': 0.01},
        'adam_beta1': {'value': 0.9},
        "adam_beta2": {'value': 0.99},
        "adam_eps": {'value': 1e-07},
        "attention_heads": {'value': 8},
        "batch_size": {'values': [128, 512, 1024]},
        "binary_preactivation": {'values': [0.5, 1.0, 10.0, 500.0]},
        "boost_function": {'values': ["GodelBoostConormApprox", "GodelBoostConorm", "LukasiewiczBoostConorm",
                                      "ProductBoostConorm"]},
        "cluster_partition_size": {'value': 8000},
        "clause_weight": {'values': [0.5, 'random', -0.5, 0.001]},
        "compliance_range": {'values': [[0.0, 1.0], [0.0, 0.2], [0.8, 1.0]]},
        "create_kb": {'value': True},
        "dataset": {'value': "PubMed"},
        'device': {'value': 0},
        "dropout": {'value': 0.5},
        "edges_drop_rate": {'values': [0.0, 0.25, 0.5, 0.9]},
        "es_enabled": {'value': True},
        "es_min_delta": {'value': 0.001},
        "es_patience": {'value': 10},
        "eval_steps": {'value': 1},
        "full_batch": {'values': [True, False]},
        "hidden_channels": {'values': [128]},
        "knowledge_base": {'value': "class_0\nCite\n\n>\n_:nclass_0(x),nCite(x.y),class_0(y)\n"},
        "knowledge_filter_key": {'value': "all"},
        "load_baseNN": {'value': False},
        "mode": {'value': "transductive"},
        "model": {'value': 'KENN_MLP'},
        "max_weight": {'values': [0.9, 1.0, 500.0]},
        "min_weight": {'values': [-0.5, 0.0]},
        "mps": {'value': False},
        "normalize_edges": {'values': [True, False]},
        "num_kenn_layers": {'values': [1, 2, 3]},
        "num_layers": {'value': 5},
        "num_layers_sampling": {'value': 3},
        "num_neighbors": {'value':[25, 10, 5, 5, 5, 5, 5, 5, 5]},
        "num_parts": {'value': 100},
        "num_steps": {'value': 5},
        "num_workers": {'value': 0},
        "planetoid_split": {'value': "full"},
        "quantity_range": {'value': [0.0, 1.0]},
        "range_constraint_lower": {'value': 0},
        "range_constraint_upper": {'value': 500},
        "runs": {'value': 1},
        "sample_coverage": {'value': 10},
        "save_data_stats": {'value': True},
        "save_results": {'value': True},
        "seed": {'value': 1234},
        "undirected": {'value': False},
        "use_node_embedding": {'value': False},
        "use_norm": {'value': False},
        "wandb_label": {'value': "label"},
        "walk_length": {'value': 3},
    }
}


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        run_experiment(args=config)


def main():
    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project="ijcai23_pubmed_kennmlp", entity="luisawerner")
    wandb.agent(sweep_id, train, count=800)


if __name__ == '__main__':
    main()
