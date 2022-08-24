import argparse
import json
import torch.multiprocessing as mp
import torch
import wandb
from experiment_config import ExperimentConf
from train_and_evaluate import run_experiment


def run_experiments(config_file):
    """
    runs a set of experiments whose parameters are described in config_file
    @param config_file: json file containing a list containing a set of parameters for each experiment
        The schema of the json is described in todo?
    """
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    with open(config_file,'r') as f:
        json_content = json.loads(f.read())
        for conf in json_content['configs']:
            experiment_conf = ExperimentConf(conf)
            wandb.init(project="test-project", entity="tyrex", config=conf)
            #run_experiment(experiment_conf)
            mp.start_processes(run_experiment, args=(world_size, experiment_conf), nprocs=world_size, join=True)
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('config', metavar='file', type=str, help='config file in json format')
    args = parser.parse_args()
    run_experiments(args.config)


if __name__ == '__main__':
    main()
