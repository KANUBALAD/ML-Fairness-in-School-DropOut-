#!/usr/bin/env python3
"""
Bulk Fairness Evaluation Script
===============================

Evaluates fairness metrics across original and synthetic datasets with cross-validation.
Updated to use YAML configurations, proper cross-validation, and consistent fairness metrics.
"""

import os
import sys
import glob
import re
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# Add the parent directory to Python path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import existing modules
import src.dataload as dataload
import src.model as model
from src.fairness import FairnessMetrics
from src.dynamic_schema_extractor import DynamicSchemaExtractor
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score

def safe_round(value, digits=4):
    """Safely round a value, handling potential None or string values."""
    try:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return round(float(value), digits)
    except (ValueError, TypeError):
        print(f"Warning: Could not round value {value}, returning as-is")
        return value

def safe_int(value):
    """Safely convert to integer."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        print(f"Warning: Could not convert {value} to int, returning 0")
        return 0

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def detect_columns(df, extractor, yaml_config):
    """
    Detect sensitive and label columns using the same logic as other updated scripts.
    """
    yaml_sensitive = yaml_config.get('sensitive_column')
    yaml_label = yaml_config.get('label_column')
    
    print(f"    YAML overrides: sensitive='{yaml_sensitive}', label='{yaml_label}'")
    
    schema = extractor.extract_comprehensive_schema(df)
    
    sens_col = yaml_sensitive or (
        schema["relationships"]["sensitive_column_candidates"][0]
        if schema["relationships"]["sensitive_column_candidates"] else None
    )
    
    label_col = yaml_label or (
        schema["relationships"]["label_column_candidates"][0]
        if schema["relationships"]["label_column_candidates"] else None
    )
    
    if sens_col is None or label_col is None:
        available_sens = schema["relationships"]["sensitive_column_candidates"]
        available_label = schema["relationships"]["label_column_candidates"]
        raise ValueError(
            f"Could not detect columns. "
            f"Available sensitive candidates: {available_sens}, "
            f"Available label candidates: {available_label}."
        )
    
    return sens_col, label_col

def determine_privileged_group(sens_counts, sens_col):
    """
    Determine which group should be considered 'privileged'.
    Returns (privileged_value, unprivileged_value) for consistent encoding.
    """
    values = sens_counts.index.tolist()
    col_lower = sens_col.lower()
    
    if 'gender' in col_lower:
        str_values = [str(v).lower() for v in values]
        
        # Standard fairness convention: Male is typically privileged
        if 'male' in str_values and 'female' in str_values:
            privileged = 'Male' if 'Male' in values else 'male'
            unprivileged = 'Female' if 'Female' in values else 'female'
        elif 'm' in str_values and 'f' in str_values:
            privileged = 'M' if 'M' in values else 'm'
            unprivileged = 'F' if 'F' in values else 'f'
        elif 0 in values and 1 in values:
            # Need to check dataset-specific encoding
            # For most datasets: 1=Male (privileged), 0=Female (unprivileged)
            privileged = 1
            unprivileged = 0
        else:
            # Default to most frequent as privileged
            privileged = sens_counts.index[0]
            unprivileged = sens_counts.index[1]
    else:
        # For other attributes, use most frequent as privileged
        privileged = sens_counts.index[0]
        unprivileged = sens_counts.index[1] if len(sens_counts.index) > 1 else sens_counts.index[0]
    
    return privileged, unprivileged

def load_data_with_sensitive(config, csv_path=None):
    """
    Load data and return features, target, and sensitive attribute using YAML configuration.
    """
    original_datapath = config['datapath']
    
    if csv_path:
        config['datapath'] = csv_path
    
    try:
        # Load features and target using existing dataload function
        X, y = dataload.load_data(config)
        
        # Load raw data for sensitive attribute extraction
        data_path = csv_path if csv_path else config['datapath']
        
        if config['dataname'] == 'india':
            data = pd.read_csv(data_path, encoding='ISO-8859-1')
        else:
            data = pd.read_csv(data_path)
        
        # Detect columns using YAML config
        extractor = DynamicSchemaExtractor()
        sens_col, label_col = detect_columns(data, extractor, config)
        
        print(f"    Detected columns: sensitive='{sens_col}', label='{label_col}'")
        
        # Verify columns exist
        if sens_col not in data.columns:
            raise ValueError(f"Sensitive column '{sens_col}' not found")
        if label_col not in data.columns:
            raise ValueError(f"Label column '{label_col}' not found")
        
        # Extract sensitive attribute
        sensitive_attr = data[sens_col].values
        
        # Get unique values and determine privileged/unprivileged groups
        unique_vals = pd.Series(sensitive_attr).value_counts()
        if len(unique_vals) != 2:
            raise ValueError(f"Sensitive attribute must have exactly 2 values, found: {unique_vals.index.tolist()}")
        
        privileged_val, unprivileged_val = determine_privileged_group(unique_vals, sens_col)
        
        # ✅ CRITICAL FIX: Ensure consistent encoding with fairness.py
        # fairness.py expects: privileged=0, unprivileged=1
        # This is OPPOSITE of typical fairness literature but matches our fairness.py implementation
        sensitive_attr_binary = (sensitive_attr == unprivileged_val).astype(int)
        
        print(f"    Loaded data: X={X.shape}, y={len(y)}, sensitive={len(sensitive_attr_binary)}")
        print(f"    Privileged group (encoded as 0): '{privileged_val}'")
        print(f"    Unprivileged group (encoded as 1): '{unprivileged_val}'")
        print(f"    Sensitive distribution: {np.bincount(sensitive_attr_binary)} (0=privileged, 1=unprivileged)")
        
        return X, y, sensitive_attr_binary
    
    except Exception as e:
        print(f"    ❌ Error loading data: {e}")
        raise
    
    finally:
        config['datapath'] = original_datapath

def parse_filename(filename):
    """Parse synthetic CSV filename to extract ratios."""
    pattern = r'.*_Sens_(\d+)_Label_(\d+)\.csv'
    match = re.match(pattern, filename)
    if match:
        sens_ratio = int(match.group(1)) / 100.0
        label_ratio = int(match.group(2)) / 100.0
        return sens_ratio, label_ratio
    return None, None

def determine_experiment_type(folder_path):
    """Determine experiment type from folder path."""
    if 'vary_sensitive' in folder_path:
        return 'vary_sensitive'
    elif 'vary_label' in folder_path:
        return 'vary_label'
    else:
        return 'unknown'

# Custom scorer functions for cross-validation
def demographic_parity_scorer(estimator, X, y, **kwargs):
    """Custom scorer for demographic parity."""
    sensitive_attr = kwargs.get('sensitive_attr', None)
    if sensitive_attr is None:
        return 0.0
    
    y_pred = estimator.predict(X)
    fairness_metrics = FairnessMetrics(y, y_pred, sensitive_attr)
    return fairness_metrics.demographic_parity()

def equalized_odds_scorer(estimator, X, y, **kwargs):
    """Custom scorer for equalized odds."""
    sensitive_attr = kwargs.get('sensitive_attr', None)
    if sensitive_attr is None:
        return 0.0
    
    y_pred = estimator.predict(X)
    fairness_metrics = FairnessMetrics(y, y_pred, sensitive_attr)
    return fairness_metrics.equalized_odds_difference()

def train_and_evaluate_model_cv(X, y, sensitive_attr, model_name, cv_folds=5, random_state=42):
    """
    Train a model with cross-validation and evaluate fairness metrics.
    
    Returns:
    dict: Dictionary with fairness metrics (mean and std)
    """
    print(f"      Training {model_name} with {cv_folds}-fold CV...")
    
    # Ensure inputs are numpy arrays
    X = np.array(X)
    y = np.array(y)
    sensitive_attr = np.array(sensitive_attr)
    
    if len(X) < 10:
        print(f"        ⚠️  Warning: Very small dataset ({len(X)} samples)")
    
    # Check for both sensitive groups
    if len(np.unique(sensitive_attr)) < 2:
        print(f"        ⚠️  Warning: Missing sensitive groups: {np.unique(sensitive_attr)}")
        return {
            'dp_mean': 0.0, 'dp_std': 0.0,
            'eo_mean': 0.0, 'eo_std': 0.0,
            'accuracy_mean': 0.0, 'accuracy_std': 0.0,
            'n': len(y)
        }
    
    try:
        # Get model
        if model_name == 'logistic_regression':
            model_func = model.logistic_regression
        elif model_name == 'decision_tree':
            model_func = model.decision_tree
        elif model_name == 'random_forest':
            model_func = model.random_forest
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Collect metrics across folds
        dp_scores = []
        eo_scores = []
        accuracy_scores = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            sens_train, sens_test = sensitive_attr[train_idx], sensitive_attr[test_idx]
            
            # Skip fold if test set doesn't have both sensitive groups
            if len(np.unique(sens_test)) < 2:
                print(f"        ⚠️  Skipping fold - missing sensitive groups in test set")
                continue
            
            # Train model
            trained_model = model_func(X_train, y_train)
            y_pred = trained_model.predict(X_test)
            
            # Calculate metrics
            fairness_metrics = FairnessMetrics(y_test, y_pred, sens_test)
            metrics = fairness_metrics.get_all_metrics()
            
            dp_scores.append(metrics['demographic_parity'])
            eo_scores.append(metrics['equalized_odds_difference'])
            accuracy_scores.append(accuracy_score(y_test, y_pred))
        
        if not dp_scores:  # No valid folds
            return {
                'dp_mean': 0.0, 'dp_std': 0.0,
                'eo_mean': 0.0, 'eo_std': 0.0,
                'accuracy_mean': 0.0, 'accuracy_std': 0.0,
                'n': len(y)
            }
        
        return {
            'dp_mean': float(np.mean(dp_scores)),
            'dp_std': float(np.std(dp_scores)),
            'eo_mean': float(np.mean(eo_scores)),
            'eo_std': float(np.std(eo_scores)),
            'accuracy_mean': float(np.mean(accuracy_scores)),
            'accuracy_std': float(np.std(accuracy_scores)),
            'n': int(len(y)),
            'cv_folds_completed': len(dp_scores)
        }
        
    except Exception as e:
        print(f"        ❌ Cross-validation failed: {e}")
        return {
            'dp_mean': 0.0, 'dp_std': 0.0,
            'eo_mean': 0.0, 'eo_std': 0.0,
            'accuracy_mean': 0.0, 'accuracy_std': 0.0,
            'n': len(y)
        }

def main():
    parser = argparse.ArgumentParser(description="Bulk fairness evaluation with cross-validation")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--folder", required=True, help="Results folder to process")
    parser.add_argument("--original", required=True, help="Original dataset CSV")
    parser.add_argument("--out_csv", required=True, help="Output CSV path")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds (default: 5)")
    
    args = parser.parse_args()
    
    print(f"=== Bulk Fairness Evaluation with Cross-Validation ===")
    print(f"Config: {args.config}")
    print(f"Folder: {args.folder}")
    print(f"Original: {args.original}")
    print(f"Output: {args.out_csv}")
    print(f"CV Folds: {args.cv_folds}")
    print()
    
    # Load config
    try:
        config = load_config(args.config)
        dataset_name = config['dataname']
        experiment_type = determine_experiment_type(args.folder)
        print(f"Dataset: {dataset_name}")
        print(f"Experiment: {experiment_type}")
        print(f"Using YAML specs: sensitive='{config.get('sensitive_column')}', label='{config.get('label_column')}'")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    models = ['logistic_regression', 'decision_tree', 'random_forest']
    results = []
    
    # 1. Evaluate on original data
    print(f"\n=== Evaluating original dataset with {args.cv_folds}-fold CV ===")
    try:
        X_orig, y_orig, sens_orig = load_data_with_sensitive(config)
        
        for model_name in models:
            print(f"  Cross-validating {model_name} on original data...")
            metrics = train_and_evaluate_model_cv(X_orig, y_orig, sens_orig, model_name, args.cv_folds)
            
            results.append({
                'csv': os.path.basename(args.original),
                'model': model_name,
                'dp_mean': safe_round(metrics['dp_mean']),
                'dp_std': safe_round(metrics['dp_std']),
                'eo_mean': safe_round(metrics['eo_mean']),
                'eo_std': safe_round(metrics['eo_std']),
                'accuracy_mean': safe_round(metrics['accuracy_mean']),
                'accuracy_std': safe_round(metrics['accuracy_std']),
                'n': safe_int(metrics['n']),
                'cv_folds': safe_int(metrics.get('cv_folds_completed', args.cv_folds)),
                'dataset': dataset_name,
                'type': 'original',
                'experiment': experiment_type,
                'sens_ratio': None,
                'label_ratio': None
            })
            
            print(f"    ✅ DP={safe_round(metrics['dp_mean'])}±{safe_round(metrics['dp_std'])}, "
                  f"EO={safe_round(metrics['eo_mean'])}±{safe_round(metrics['eo_std'])}, "
                  f"Acc={safe_round(metrics['accuracy_mean'])}±{safe_round(metrics['accuracy_std'])}")
            
    except Exception as e:
        print(f"  ❌ Error processing original data: {e}")
    
    # 2. Evaluate on synthetic data
    print(f"\n=== Evaluating synthetic datasets with {args.cv_folds}-fold CV ===")
    
    csv_pattern = os.path.join(args.folder, "*.csv")
    synthetic_files = glob.glob(csv_pattern)
    
    # Filter synthetic files
    synthetic_files = [f for f in synthetic_files if not any(x in os.path.basename(f).lower() for x in 
                      ['analysis', 'fairness_bulk_results', 'joint_distribution', 'sensitive_attribute'])]
    
    dataset_pattern = f"{dataset_name}_Sens_"
    synthetic_files = [f for f in synthetic_files if dataset_pattern in os.path.basename(f)]
    synthetic_files.sort()
    
    print(f"Found {len(synthetic_files)} synthetic files")
    
    for i, csv_file in enumerate(synthetic_files, 1):
        filename = os.path.basename(csv_file)
        print(f"  [{i}/{len(synthetic_files)}] Processing {filename}...")
        
        sens_ratio, label_ratio = parse_filename(filename)
        if sens_ratio is None or label_ratio is None:
            print(f"    ⚠️  Could not parse ratios, skipping")
            continue
        
        try:
            X_syn, y_syn, sens_syn = load_data_with_sensitive(config, csv_file)
            
            for model_name in models:
                metrics = train_and_evaluate_model_cv(X_syn, y_syn, sens_syn, model_name, args.cv_folds)
                
                results.append({
                    'csv': filename,
                    'model': model_name,
                    'dp_mean': safe_round(metrics['dp_mean']),
                    'dp_std': safe_round(metrics['dp_std']),
                    'eo_mean': safe_round(metrics['eo_mean']),
                    'eo_std': safe_round(metrics['eo_std']),
                    'accuracy_mean': safe_round(metrics['accuracy_mean']),
                    'accuracy_std': safe_round(metrics['accuracy_std']),
                    'n': safe_int(metrics['n']),
                    'cv_folds': safe_int(metrics.get('cv_folds_completed', args.cv_folds)),
                    'dataset': dataset_name,
                    'type': 'synthetic',
                    'experiment': experiment_type,
                    'sens_ratio': safe_round(sens_ratio),
                    'label_ratio': safe_round(label_ratio)
                })
            
            print(f"    ✅ Processed {len(models)} models with CV")
                
        except Exception as e:
            print(f"    ❌ Error processing {filename}: {e}")
    
    # 3. Save results
    print(f"\n=== Saving results ===")
    
    if not results:
        print("❌ No results to save!")
        return 1
    
    try:
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        results_df.to_csv(args.out_csv, index=False, float_format='%.4f')
        
        print(f"✅ Completed! {len(results)} results saved to {args.out_csv}")
        print(f"   Original samples: {len([r for r in results if r['type'] == 'original'])}")
        print(f"   Synthetic samples: {len([r for r in results if r['type'] == 'synthetic'])}")
        
        # Summary statistics
        if results:
            print(f"\nSummary by model (original data):")
            for model_name in models:
                orig_results = results_df[(results_df['model'] == model_name) & (results_df['type'] == 'original')]
                if len(orig_results) > 0:
                    row = orig_results.iloc[0]
                    print(f"  {model_name}: Acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}, "
                          f"DP={row['dp_mean']:.3f}±{row['dp_std']:.3f}, "
                          f"EO={row['eo_mean']:.3f}±{row['eo_std']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)