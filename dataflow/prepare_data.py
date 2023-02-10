import os
import argparse
from dataflow.utils import read_yaml
from dataflow.importers.timit import TimitImporter
from dataflow.importers.common_voice import CVImporter
from dataflow.importers.mozilla import MozillaCVImporter


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', 
                        type=str,
                        required=True)
    return parser.parse_args()


name_to_class = {"timit": TimitImporter,
                 'cv': CVImporter,
                 'mozilla': MozillaCVImporter}

if __name__ == '__main__':

    parser = arg_parser()
    config = read_yaml(parser.config_path)
    datasets_to_process = config["datasets_to_process"]

    os.makedirs(config['target_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['target_dir'], "labels"), exist_ok=True)

    for dataset_name in datasets_to_process:
        dataset_importer = name_to_class[dataset_name](config)
        print(dataset_importer)
        # dataset_importer.import_dataset()
