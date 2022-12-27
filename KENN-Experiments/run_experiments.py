import argparse
import json
import wandb
from train_and_evaluate import run_experiment


class ExperimentConf(object):
    def __init__(self, conf_dict):
        for key, value in conf_dict.items():
            setattr(self, key, value)


def run_experiments(config_file):
    """
    runs a set of experiments whose parameters are described in config_file
    @param config_file: json file containing a list containing a set of parameters for each experiment
        The schema of the json is described in todo?
    """

    with open(config_file, 'r') as f:
        json_content = json.loads(f.read())
    for conf in json_content['configs']:
        experiment_conf = ExperimentConf(conf)
        wandb.init(project="ijcai23", entity="luisawerner", tags=[experiment_conf.wandb_label], config=conf, mode='disabled')
        run_experiment(experiment_conf)
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('config', metavar='file', type=str, help='config file in json format')
    args = parser.parse_args()
    run_experiments(args.config)


if __name__ == '__main__':
    main()
