
class ExperimentConf(object):
    def __init__(self, conf_dict):
        for key, value in conf_dict.items():
            setattr(self, key, value)


        # self.dataset = conf_dict['dataset']
        # self.planetoid_split = conf_dict['planetoid_split']
        # self.device = conf_dict['device']
        # self.use_norm = conf_dict['use_norm']
        # self.use_node_embedding = conf_dict['use_node_embedding']
        # self.num_layers = conf_dict['num_layers']
        # self.num_layers_sampling = conf_dict['num_layers_sampling']
        # self.hidden_channels = conf_dict['hidden_channels']
        # self.dropout = conf_dict['dropout']
        # self.lr = conf_dict['lr']
        # self.epochs = conf_dict['epochs']
        # self.runs = conf_dict['runs']
        # self.model = conf_dict['model']
        # self.mode = conf_dict['mode']
        # self.save_results = conf_dict['save_results']
        # self.binary_preactivation = conf_dict['binary_preactivation']
        # self.num_kenn_layers = conf_dict['num_kenn_layers']
        # self.range_constraint_lower = conf_dict['range_constraint_lower']
        # self.range_constraint_upper = conf_dict['range_constraint_upper']
        # self.es_enabled = conf_dict['es_enabled']
        # self.es_min_delta = conf_dict['es_min_delta']
        # self.es_patience = conf_dict['es_patience']
        # self.sampling_neighbor_size = conf_dict['sampling_neighbor_size']
        # self.batch_size = conf_dict['batch_size']
        # self.full_batch = conf_dict['full_batch']
        # self.num_workers = conf_dict['num_workers']
        # self.seed = conf_dict['seed']
        # self.train_sampling = conf_dict['train_sampling']
        # self.cluster_partition_size = conf_dict['cluster_partition_size']
        # self.sample_coverage = conf_dict['sample_coverage']
        # self.walk_length = conf_dict['walk_length']
        # self.num_steps = conf_dict['num_steps']
        # self.eval_steps = conf_dict['eval_steps']
        # self.save_data_stats = conf_dict['save_data_stats']
        # self.create_kb = conf_dict['create_kb']
        # self.knowledge_base = conf_dict['knowledge_base']
        # self.num_parts = conf_dict['num_parts']
        # self.normalize_edges = conf_dict['normalize_edges']
        # self.attention_heads = conf_dict['attention_heads']
