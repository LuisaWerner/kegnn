
class ExperimentConf(object):
    def __init__(self, conf_dict):
        self.dataset = conf_dict['dataset']
        self.device = conf_dict['device']
        self.log_steps = conf_dict['log_steps']
        self.use_node_embedding = conf_dict['use_node_embedding']
        self.num_layers = conf_dict['num_layers']
        self.num_layers_sampling = conf_dict['num_layers_sampling']
        self.hidden_channels = conf_dict['hidden_channels']
        self.dropout = conf_dict['dropout']
        self.lr = conf_dict['lr']
        self.epochs = conf_dict['epochs']
        self.runs = conf_dict['runs']
        self.model = conf_dict['model']
        self.mode = conf_dict['mode']
        self.save_results = conf_dict['save_results']
        self.binary_preactivation = conf_dict['binary_preactivation']
        self.num_kenn_layers = conf_dict['num_kenn_layers']
        self.range_constraint_lower = conf_dict['range_constraint_lower']
        self.range_constraint_upper = conf_dict['range_constraint_upper']
        self.es_enabled = conf_dict['es_enabled']
        self.es_min_delta = conf_dict['es_min_delta']
        self.es_patience = conf_dict['es_patience']
        self.sampling_neighbor_size = conf_dict['sampling_neighbor_size']
        self.batch_size = conf_dict['batch_size']
        self.full_batch = conf_dict['full_batch']
        self.num_workers = conf_dict['num_workers']
        self.seed = conf_dict['seed']

