import os
import yaml
import argparse
import numpy as np
import pandas as pd

from src import utils, dataload, model
import src.fairness as fairness

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    try:
        print(f"Current working directory: {os.getcwd()}")
        config = load_config(config_path)
        print(f"Loading configuration from: {config_path} and fairness unawareness is {config['unawareness']}")

        # Hyperparameters and switches
        fairness_on = config.get('fairness', False)
        fair_technique = config.get('fair_technique', 'reweighting')
        which_model = config.get('model', 'logistic_regression')
        testsize = config['test_size']
        randomstate = config['random_state']

        # Load preprocessed features and labels
        Xtransformed, ytarget = dataload.load_data(config)

        # Read the original data for sensitive attribute extraction
        raw_data = pd.read_csv(config['datapath'])
        sensitive_attr = utils.extract_sensitive_attribute(raw_data, config)
        print(f"Sensitive attribute (privileged=1, unprivileged=0) counts: {np.bincount(sensitive_attr)}")

        # Split the data, including the sensitive variable
        split = utils.split_data(np.array(Xtransformed), np.array(ytarget), sens=sensitive_attr,
                                 test_size=testsize, random_state=randomstate)
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']
        sens_train, sens_test = split['sens_train'], split['sens_test']

        print(f"Xtrain shape: {X_train.shape}, ytrain shape: {y_train.shape}")
        print(f"Xtest shape: {X_test.shape}, ytest shape: {y_test.shape}")

        # Standard ML (with cross-val comparison)
        print("Performing 5x2 cross-validation...")
        cross_val_results = model.run_cross_validation(config, np.array(Xtransformed), np.array(ytarget))
        for model_name, metrics in cross_val_results.items():
            print(f"{model_name}: Mean accuracy = {metrics['mean']:.4f}, Std deviation = {metrics['std']:.4f}")

        # Baseline ML model for direct fairness evaluation (use logistic regression, tree, or all)
        base_models = []
        if which_model == 'compare':
            base_models = ['logistic_regression', 'decision_tree', 'random_forest']
        else:
            base_models = [which_model]

        for m in base_models:
            if m == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=1000, random_state=randomstate)
            elif m == 'decision_tree':
                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier(random_state=randomstate)
            elif m == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(random_state=randomstate)
            else:
                print(f"Unknown model: {m}, skipping.")
                continue

            clf.fit(X_train, y_train)
            y_pred_baseline = clf.predict(X_test)
            metrics_base = utils.fairness_summary(y_pred_baseline, y_test, sens_test, model_name=f"Baseline {m}")
            utils.print_fairness_report(metrics_base)

            # Fair ML if enabled
            if fairness_on:
                print(f"\nRunning fairness mitigation ({fair_technique}) using model: {m}")
                fair_out = fairness.run_fairness_aware_training(
                    np.array(Xtransformed), np.array(ytarget), sensitive_attr,
                    model_type=m, 
                    technique=fair_technique, 
                    test_size=testsize, random_state=randomstate)
                print("=== Fair Model results ===")
                utils.print_fairness_report(
                    utils.fairness_summary(
                        fair_out['y_pred_fair'], fair_out['y_test'], fair_out['sens_test'], model_name=fair_out['fair_metrics'].get('model_name','Fair Model')
                    )
                )

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training and evaluation pipeline.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    # Optional CLI fairness switches (override YAML)
    parser.add_argument("--fairness", action='store_true', help="Whether to run fairness-aware training.")
    parser.add_argument("--fair_technique", type=str, default=None, help="Fairness technique to use (reweighting, fair_representation, etc)")
    args = parser.parse_args()

    config = load_config(args.config_path)
    if args.fairness:
        config['fairness'] = True
    if args.fair_technique is not None:
        config['fair_technique'] = args.fair_technique
    main(args.config_path)