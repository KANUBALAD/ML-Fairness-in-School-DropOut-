import os
import yaml
from src import utlis,dataload
import argparse



def load_config(config_path):
    """
    Load configuration from a YAML file.

    Parameters:
    config_path (str): Path to the configuration file.

    Returns:
    dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config_path):
    """
    Main function to execute the training and evaluation pipeline.
    Parameters:
    config_path (str): Path to the configuration file.
    """
    try:
        print(f"Current working directory: {os.getcwd()}")
        config = load_config(config_path)
        print(f"Loading configuration from: {config_path} and fairness unawarness is {config['unawareness']}")

        testsize = config['test_size']
        randomstate = config['random_state']
        Xtransformed, ytarget = dataload.load_data(config)
        split_data = utlis.split_data(Xtransformed, ytarget, testsize, randomstate)
        print(f"Xtrain has shape {split_data['X_train'].shape} and ytrain has shape {split_data['y_train'].shape}")
        print(f"Xtest has shape {split_data['X_test'].shape} and ytest has shape {split_data['y_test'].shape}")    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training and evaluation pipeline.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config_path)