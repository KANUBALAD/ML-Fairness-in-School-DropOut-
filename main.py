import os
import yaml
import argparse
import numpy as np
import pandas as pd
import json
import asyncio
from datetime import datetime

from src import utils, dataload, model
import src.fairness as fairness
from src.synthetic_generator import SyntheticDataGenerator

# Import new generators
try:
    import src.llm_synthetic_generator as llm_gen
    LLM_AVAILABLE = True
    print("✓ LLM generator loaded successfully")
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"LLM generator not available: {e}")

try:
    from src.CTGAN_TVAE_synthetic_generator import run_ctgan_experiment, run_tvae_experiment
    CTGAN_AVAILABLE = True
    print("✓ CTGAN/TVAE generators loaded successfully")
except ImportError as e:
    CTGAN_AVAILABLE = False
    print(f"CTGAN/TVAE generators not available: {e}")

def create_results_folder():
    """Create a dedicated results folder with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"./results/experiment_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    return results_folder

def save_results_to_json(results, results_folder, experiment_type, config_name, scenario_name=None):
    """Save experiment results to JSON file."""
    
    # Create filename
    if scenario_name:
        filename = f"{experiment_type}_{config_name}_{scenario_name}.json"
    else:
        filename = f"{experiment_type}_{config_name}.json"
    
    filepath = os.path.join(results_folder, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Add metadata
    results_with_metadata = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': experiment_type,
            'config_name': config_name,
            'scenario_name': scenario_name
        },
        'results': convert_numpy_types(results)
    }
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(results_with_metadata, f, indent=2, default=str)
    
    print(f"✓ Saved results to: {filepath}")
    return filepath

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def analyze_dataset_imbalance(data, config):
    """Analyze current imbalance in the dataset."""
    
    # Get sensitive attribute and label columns
    if config['dataname'] == 'africa':
        sensitive_col = 'gender'
        label_col = 'dropout'
        privileged_val = 'Male'
        positive_label = 'Yes'
    elif config['dataname'] == 'brazil':
        sensitive_col = 'Gender'
        label_col = 'Target'
        privileged_val = 0  # Male
        positive_label = 'Dropout'
    elif config['dataname'] == 'india':
        sensitive_col = 'STUDENTGENDER'
        label_col = 'STUDENT_DROPOUT_STATUS'
        privileged_val = 'M'
        positive_label = 'DROPOUT'
    
    # Count current distributions
    total_samples = len(data)
    
    # Sensitive attribute distribution
    privileged_count = sum(data[sensitive_col] == privileged_val)
    unprivileged_count = total_samples - privileged_count
    current_sensitive_ratio = privileged_count / total_samples
    
    # Label distribution
    positive_count = sum(data[label_col] == positive_label)
    negative_count = total_samples - positive_count
    current_label_ratio = positive_count / total_samples
    
    # Cross-tabulation
    priv_pos = sum((data[sensitive_col] == privileged_val) & (data[label_col] == positive_label))
    priv_neg = privileged_count - priv_pos
    unpriv_pos = positive_count - priv_pos
    unpriv_neg = unprivileged_count - unpriv_pos
    
    analysis = {
        'total_samples': total_samples,
        'privileged_count': privileged_count,
        'unprivileged_count': unprivileged_count,
        'current_sensitive_ratio': current_sensitive_ratio,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'current_label_ratio': current_label_ratio,
        'cross_tab': {
            'priv_pos': priv_pos,
            'priv_neg': priv_neg,
            'unpriv_pos': unpriv_pos,
            'unpriv_neg': unpriv_neg
        },
        'columns': {
            'sensitive_col': sensitive_col,
            'label_col': label_col,
            'privileged_val': privileged_val,
            'positive_label': positive_label
        }
    }
    
    return analysis

def calculate_augmentation_needs(analysis, target_sensitive_ratio, target_label_ratio):
    """Calculate how many additional samples of each type are needed."""
    
    current_total = analysis['total_samples']
    current_priv = analysis['privileged_count']
    current_unpriv = analysis['unprivileged_count']
    current_pos = analysis['positive_count']
    current_neg = analysis['negative_count']
    
    # Calculate minimum total size needed for sensitive attribute balance
    if target_sensitive_ratio == 0.5:
        min_total_for_sensitive = 2 * max(current_priv, current_unpriv)
    else:
        if current_priv / current_total < target_sensitive_ratio:
            min_total_for_sensitive = current_priv / target_sensitive_ratio
        else:
            min_total_for_sensitive = current_unpriv / (1 - target_sensitive_ratio)
    
    # Calculate minimum total size needed for label balance
    if target_label_ratio == 0.5:
        min_total_for_labels = 2 * max(current_pos, current_neg)
    else:
        if current_pos / current_total < target_label_ratio:
            min_total_for_labels = current_pos / target_label_ratio
        else:
            min_total_for_labels = current_neg / (1 - target_label_ratio)
    
    # Use the larger of the two minimum totals
    target_total = max(min_total_for_sensitive, min_total_for_labels, current_total)
    target_total = int(np.ceil(target_total))
    
    # Calculate target counts
    target_priv = int(target_total * target_sensitive_ratio)
    target_unpriv = target_total - target_priv
    target_pos = int(target_total * target_label_ratio)
    target_neg = target_total - target_pos
    
    # Calculate cross-combinations needed
    target_priv_pos = int(target_priv * target_label_ratio)
    target_priv_neg = target_priv - target_priv_pos
    target_unpriv_pos = target_pos - target_priv_pos
    target_unpriv_neg = target_unpriv - target_unpriv_pos
    
    # Calculate additional samples needed
    current_cross = analysis['cross_tab']
    
    additional_needed = {
        'total_additional': target_total - current_total,
        'target_total': target_total,
        'breakdown': {
            'priv_pos': max(0, target_priv_pos - current_cross['priv_pos']),
            'priv_neg': max(0, target_priv_neg - current_cross['priv_neg']),
            'unpriv_pos': max(0, target_unpriv_pos - current_cross['unpriv_pos']),
            'unpriv_neg': max(0, target_unpriv_neg - current_cross['unpriv_neg'])
        },
        'target_distribution': {
            'priv_pos': target_priv_pos,
            'priv_neg': target_priv_neg,
            'unpriv_pos': target_unpriv_pos,
            'unpriv_neg': target_unpriv_neg
        }
    }
    
    return additional_needed

async def run_llm_async_generation(original_data, config, api_key, augmentation_plan):
    """
    Run LLM async generation using the available functions in llm_synthetic_generator.
    """
    
    print("🤖 Starting LLM async generation...")
    
    try:
        # Check if generate_enhanced_synthetic_data function exists
        if hasattr(llm_gen, 'generate_enhanced_synthetic_data'):
            print("✓ Using generate_enhanced_synthetic_data function")
            augmented_data = await llm_gen.generate_enhanced_synthetic_data(
                original_data, config, api_key, augmentation_plan
            )
            return augmented_data
        
        # Check if AsyncLLMSyntheticGenerator class exists
        elif hasattr(llm_gen, 'AsyncLLMSyntheticGenerator'):
            print("✓ Using AsyncLLMSyntheticGenerator class")
            generator = llm_gen.AsyncLLMSyntheticGenerator(api_key, max_concurrent=2)
            
            # Prepare target specs from augmentation plan
            breakdown = augmentation_plan['breakdown']
            target_specs = []
            
            # Map categories to actual column values
            if config['dataname'] == 'brazil':
                mapping = {'priv_pos': (1, 'Dropout'), 'priv_neg': (1, 'Graduate'), 
                          'unpriv_pos': (0, 'Dropout'), 'unpriv_neg': (0, 'Graduate')}
            elif config['dataname'] == 'africa':
                mapping = {'priv_pos': ('Male', 'Yes'), 'priv_neg': ('Male', 'No'), 
                          'unpriv_pos': ('Female', 'Yes'), 'unpriv_neg': ('Female', 'No')}
            elif config['dataname'] == 'india':
                mapping = {'priv_pos': ('M', 'DROPOUT'), 'priv_neg': ('M', 'NOT DROPOUT'), 
                          'unpriv_pos': ('F', 'DROPOUT'), 'unpriv_neg': ('F', 'NOT DROPOUT')}
            
            for category, count in breakdown.items():
                if count > 0:
                    sensitive_val, label_val = mapping[category]
                    target_specs.append({
                        'category': category,
                        'count': count,
                        'sensitive_val': sensitive_val,
                        'label_val': label_val
                    })
            
            # Generate samples
            synthetic_samples = await generator.generate_conditional_samples_batch(
                config['dataname'], target_specs, original_data
            )
            
            if synthetic_samples:
                synthetic_df = pd.DataFrame(synthetic_samples)
                
                # Ensure column compatibility
                for col in original_data.columns:
                    if col not in synthetic_df.columns:
                        synthetic_df[col] = 'Unknown'
                
                # Combine with original
                augmented_data = pd.concat([original_data, synthetic_df], ignore_index=True)
                return augmented_data
            else:
                return original_data
        
        else:
            print("⚠ No compatible LLM generation functions found")
            return original_data
            
    except Exception as e:
        print(f"✗ LLM generation failed: {e}")
        return original_data

def generate_targeted_synthetic_samples(original_data, config, api_key, augmentation_plan, method='faker'):
    """
    Generate specific synthetic samples to achieve target balance using specified method.
    """
    
    print(f"\nGenerating targeted synthetic samples using method: {method}")
    print(f"Total additional samples needed: {augmentation_plan['total_additional']}")
    
    breakdown = augmentation_plan['breakdown']
    
    # CTGAN Method
    if method == 'ctgan':
        if not CTGAN_AVAILABLE:
            print("Error: CTGAN not available. Install with: pip install sdv")
            print("Falling back to Faker method...")
            method = 'faker'
        else:
            print("Using CTGAN generator...")
            try:
                augmented_data = run_ctgan_experiment(original_data, config, augmentation_plan)
                return augmented_data
            except Exception as e:
                print(f"CTGAN generation failed: {e}")
                print("Falling back to Faker method...")
                method = 'faker'
    
    # TVAE Method
    elif method == 'tvae':
        if not CTGAN_AVAILABLE:
            print("Error: TVAE not available. Install with: pip install sdv")
            print("Falling back to Faker method...")
            method = 'faker'
        else:
            print("Using TVAE generator...")
            try:
                augmented_data = run_tvae_experiment(original_data, config, augmentation_plan)
                return augmented_data
            except Exception as e:
                print(f"TVAE generation failed: {e}")
                print("Falling back to Faker method...")
                method = 'faker'
    
    # LLM Async Method
    elif method == 'llm_async':
        if not LLM_AVAILABLE:
            print("Error: LLM generator not available")
            print("Falling back to Faker method...")
            method = 'faker'
        elif not api_key or api_key == 'dummy':
            print("Warning: No valid API key provided for LLM method")
            print("Falling back to Faker method...")
            method = 'faker'
        else:
            print("Using LLM async generator...")
            try:
                # Run the async LLM generation
                augmented_data = asyncio.run(run_llm_async_generation(
                    original_data, config, api_key, augmentation_plan
                ))
                return augmented_data
            except Exception as e:
                print(f"LLM generation failed: {e}")
                print("Falling back to Faker method...")
                method = 'faker'
    
    # Faker Method (default fallback)
    if method == 'faker':
        print("Using Faker generator...")
        generator = SyntheticDataGenerator()
        
        # Generate samples for each category
        all_synthetic_samples = []
        
        categories = [
            ('priv_pos', 'Privileged + Positive'),
            ('priv_neg', 'Privileged + Negative'), 
            ('unpriv_pos', 'Unprivileged + Positive'),
            ('unpriv_neg', 'Unprivileged + Negative')
        ]
        
        for category_key, category_name in categories:
            needed_count = breakdown[category_key]
            
            if needed_count > 0:
                print(f"Generating {needed_count} samples for {category_name}")
                
                # Generate samples for this specific category
                category_samples = generator.generate_category_specific_samples(
                    dataset_name=config['dataname'],
                    n_samples=needed_count,
                    category=category_key,
                    reference_data=original_data
                )
                
                all_synthetic_samples.extend(category_samples)
        
        # Convert to DataFrame
        if all_synthetic_samples:
            synthetic_df = pd.DataFrame(all_synthetic_samples)
            
            # Ensure column compatibility with original data
            for col in original_data.columns:
                if col not in synthetic_df.columns:
                    # Generate appropriate default values
                    synthetic_df[col] = generator._generate_default_column(col, config['dataname'], len(synthetic_df))
            
            # Reorder columns to match original
            synthetic_df = synthetic_df.reindex(columns=original_data.columns, fill_value='Unknown')
            
            # Combine with original data
            augmented_data = pd.concat([original_data, synthetic_df], ignore_index=True)
        else:
            print("No additional samples needed!")
            augmented_data = original_data.copy()
        
        return augmented_data

def test_fairness_on_balanced_data(augmented_data, config, scenario_name, original_analysis, final_analysis, target_sensitive_ratio, target_label_ratio, method='faker'):
    """Test fairness interventions on the balanced augmented dataset."""
    
    print(f"\nTesting fairness interventions on balanced dataset (method: {method})...")
    
    # Create temporary config for augmented data
    temp_path = f"./data/temp_augmented_{config['dataname']}_{scenario_name}_{method}.csv"
    augmented_data.to_csv(temp_path, index=False)
    temp_config = config.copy()
    temp_config['datapath'] = temp_path
    
    try:
        # Load and preprocess augmented data
        X_transformed, y_target = dataload.load_data(temp_config)
        sensitive_attr = utils.extract_sensitive_attribute(augmented_data, temp_config)
        
        print(f"Preprocessed balanced data shape: {X_transformed.shape}")
        print(f"Final sensitive distribution: {np.bincount(sensitive_attr)}")
        print(f"Final label distribution: {np.bincount(y_target)}")
        
        # Split the data
        split = utils.split_data(np.array(X_transformed), np.array(y_target), 
                               sens=sensitive_attr, test_size=config['test_size'], 
                               random_state=config['random_state'])
        
        X_train, X_test = split['X_train'], split['X_test']
        y_train, y_test = split['y_train'], split['y_test']
        sens_train, sens_test = split['sens_train'], split['sens_test']
        
        # Test baseline model on balanced data
        from sklearn.linear_model import LogisticRegression
        baseline_model = LogisticRegression(max_iter=1000, random_state=config['random_state'])
        baseline_model.fit(X_train, y_train)
        y_pred_baseline = baseline_model.predict(X_test)
        
        baseline_metrics = utils.fairness_summary(y_pred_baseline, y_test, sens_test, 
                                                model_name=f"Baseline_Balanced_{scenario_name}_{method}")
        
        print(f"\n--- Baseline Model on Balanced Data ({method}) ---")
        utils.print_fairness_report(baseline_metrics)
        
        # Test fairness intervention if enabled
        fair_metrics = None
        if config.get('fairness', False):
            print(f"\nApplying fairness intervention on balanced data: {config.get('fair_technique', 'reweighting')}")
            
            fair_results = fairness.run_fairness_aware_training(
                np.array(X_transformed), np.array(y_target), sensitive_attr,
                model_type='logistic_regression',
                technique=config.get('fair_technique', 'reweighting'),
                test_size=config['test_size'],
                random_state=config['random_state']
            )
            
            fair_metrics = utils.fairness_summary(
                fair_results['y_pred_fair'], fair_results['y_test'], 
                fair_results['sens_test'], model_name=f"Fair_Balanced_{scenario_name}_{method}"
            )
            
            print(f"\n--- Fair Model on Balanced Data ({method}) ---")
            utils.print_fairness_report(fair_metrics)
        
        # Compile results
        results = {
            'scenario_name': scenario_name,
            'generation_method': method,
            'original_analysis': original_analysis,
            'final_analysis': final_analysis,
            'augmented_data_shape': list(augmented_data.shape),
            'target_ratios': {
                'target_sensitive_ratio': target_sensitive_ratio,
                'target_label_ratio': target_label_ratio
            },
            'baseline_metrics': baseline_metrics,
            'fair_metrics': fair_metrics,
            'balance_achieved': {
                'sensitive_ratio_achieved': abs(final_analysis['current_sensitive_ratio'] - target_sensitive_ratio) < 0.05,
                'label_ratio_achieved': abs(final_analysis['current_label_ratio'] - target_label_ratio) < 0.05
            }
        }
        
        # Calculate improvement metrics
        if fair_metrics:
            results['improvement_metrics'] = {
                'accuracy_difference': baseline_metrics['overall_accuracy'] - fair_metrics['overall_accuracy'],
                'dp_improvement': (abs(baseline_metrics.get('demographic_parity_difference', 0)) - 
                                 abs(fair_metrics.get('demographic_parity_difference', 0))),
                'tpr_improvement': (abs(baseline_metrics.get('tpr_difference', 0)) - 
                                  abs(fair_metrics.get('tpr_difference', 0))),
                'fpr_improvement': (abs(baseline_metrics.get('fpr_difference', 0)) - 
                                  abs(fair_metrics.get('fpr_difference', 0)))
            }
        
        return results
        
    except Exception as e:
        print(f"Error testing fairness on balanced data: {e}")
        return {'error': str(e), 'scenario_name': scenario_name, 'generation_method': method}
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

def run_balanced_experiment(config_path, api_key, target_sensitive_ratio, target_label_ratio, scenario_name="balanced_experiment", results_folder=None, method='faker'):
    """Main function to run balanced augmentation experiment."""
    
    # Load configuration
    config = load_config(config_path)
    
    print(f"\n{'='*80}")
    print(f"BALANCED AUGMENTATION EXPERIMENT: {scenario_name}")
    print(f"Dataset: {config['dataname']}")
    print(f"Generation Method: {method}")
    print(f"Target Sensitive Ratio: {target_sensitive_ratio:.2f} (privileged:unprivileged)")
    print(f"Target Label Ratio: {target_label_ratio:.2f} (positive:negative)")
    print(f"{'='*80}")
    
    # Load original data
    original_data = pd.read_csv(config['datapath'])
    print(f"Original dataset shape: {original_data.shape}")
    
    # Analyze current imbalance
    analysis = analyze_dataset_imbalance(original_data, config)
    
    print(f"\nCurrent Distribution Analysis:")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Privileged: {analysis['privileged_count']} ({analysis['current_sensitive_ratio']:.3f})")
    print(f"  Unprivileged: {analysis['unprivileged_count']} ({1-analysis['current_sensitive_ratio']:.3f})")
    print(f"  Positive labels: {analysis['positive_count']} ({analysis['current_label_ratio']:.3f})")
    print(f"  Negative labels: {analysis['negative_count']} ({1-analysis['current_label_ratio']:.3f})")
    
    print(f"\nCross-tabulation:")
    cross_tab = analysis['cross_tab']
    print(f"  Privileged + Positive: {cross_tab['priv_pos']}")
    print(f"  Privileged + Negative: {cross_tab['priv_neg']}")
    print(f"  Unprivileged + Positive: {cross_tab['unpriv_pos']}")
    print(f"  Unprivileged + Negative: {cross_tab['unpriv_neg']}")
    
    # Calculate augmentation needs
    augmentation_plan = calculate_augmentation_needs(analysis, target_sensitive_ratio, target_label_ratio)
    
    print(f"\nAugmentation Plan:")
    print(f"  Total additional samples needed: {augmentation_plan['total_additional']}")
    print(f"  Target total size: {augmentation_plan['target_total']}")
    
    breakdown = augmentation_plan['breakdown']
    if any(breakdown.values()):
        print(f"  Additional samples by category:")
        print(f"    Privileged + Positive: +{breakdown['priv_pos']}")
        print(f"    Privileged + Negative: +{breakdown['priv_neg']}")
        print(f"    Unprivileged + Positive: +{breakdown['unpriv_pos']}")
        print(f"    Unprivileged + Negative: +{breakdown['unpriv_neg']}")
    else:
        print(f"  No additional samples needed - dataset already meets target ratios!")
        results = {
            'scenario_name': scenario_name,
            'generation_method': method,
            'original_analysis': analysis,
            'final_analysis': analysis,
            'augmented_data_shape': list(original_data.shape),
            'balance_achieved': True,
            'message': 'No augmentation needed',
            'target_ratios': {
                'target_sensitive_ratio': target_sensitive_ratio,
                'target_label_ratio': target_label_ratio
            }
        }
        
        if results_folder:
            save_results_to_json(results, results_folder, f'balanced_augmentation_{method}', 
                                config['dataname'], scenario_name)
        
        return results
    
    # Generate augmented dataset
    config['columns'] = analysis['columns']  # Pass column info to config
    augmented_data = generate_targeted_synthetic_samples(original_data, config, api_key, augmentation_plan, method)
    
    print(f"\nAugmented dataset shape: {augmented_data.shape}")
    
    # Verify the balance
    verify_analysis = analyze_dataset_imbalance(augmented_data, config)
    print(f"\nVerification - Final Distribution:")
    print(f"  Privileged ratio: {verify_analysis['current_sensitive_ratio']:.3f} (target: {target_sensitive_ratio:.3f})")
    print(f"  Positive label ratio: {verify_analysis['current_label_ratio']:.3f} (target: {target_label_ratio:.3f})")
    
    # Save augmented dataset
    output_path = f"./data/augmented_{config['dataname']}_{scenario_name}_{method}_{target_sensitive_ratio}_{target_label_ratio}.csv"
    augmented_data.to_csv(output_path, index=False)
    print(f"✓ Saved augmented dataset to: {output_path}")
    
    # Test fairness on augmented data
    results = test_fairness_on_balanced_data(augmented_data, config, scenario_name, analysis, verify_analysis, target_sensitive_ratio, target_label_ratio, method)
    
    # Add augmentation plan to results
    results['augmentation_plan'] = augmentation_plan
    results['augmented_dataset_path'] = output_path
    
    # Save results if folder provided
    if results_folder:
        save_results_to_json(results, results_folder, f'balanced_augmentation_{method}', 
                            config['dataname'], scenario_name)
    
    return results

def run_standard_ml_experiment(config_path, results_folder=None):
    """Run standard ML pipeline and save results."""
    
    config = load_config(config_path)
    
    print(f"\n{'='*80}")
    print(f"STANDARD ML EXPERIMENT")
    print(f"Dataset: {config['dataname']}")
    print(f"{'='*80}")
    
    # Initialize results dictionary
    experiment_results = {
        'config': config,
        'cross_validation_results': {},
        'model_results': {}
    }
    
    try:
        print(f"Current working directory: {os.getcwd()}")
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

        # Add data analysis to results
        experiment_results['data_analysis'] = {
            'total_samples': len(Xtransformed),
            'features_shape': Xtransformed.shape,
            'sensitive_distribution': np.bincount(sensitive_attr).tolist(),
            'label_distribution': np.bincount(ytarget).tolist()
        }

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
        experiment_results['cross_validation_results'] = cross_val_results
        
        for model_name, metrics in cross_val_results.items():
            print(f"{model_name}: Mean accuracy = {metrics['mean']:.4f}, Std deviation = {metrics['std']:.4f}")

        # Baseline ML model for direct fairness evaluation
        base_models = []
        if which_model == 'compare':
            base_models = ['logistic_regression', 'decision_tree', 'random_forest']
        else:
            base_models = [which_model]

        for m in base_models:
            print(f"\nEvaluating {m}...")
            
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
            
            # Store baseline results
            experiment_results['model_results'][m] = {
                'baseline_metrics': metrics_base
            }

            # Fair ML if enabled
            if fairness_on:
                print(f"\nRunning fairness mitigation ({fair_technique}) using model: {m}")
                fair_out = fairness.run_fairness_aware_training(
                    np.array(Xtransformed), np.array(ytarget), sensitive_attr,
                    model_type=m, 
                    technique=fair_technique, 
                    test_size=testsize, random_state=randomstate)
                
                fair_metrics = utils.fairness_summary(
                    fair_out['y_pred_fair'], fair_out['y_test'], fair_out['sens_test'], 
                    model_name=fair_out['fair_metrics'].get('model_name','Fair Model')
                )
                
                print("=== Fair Model results ===")
                utils.print_fairness_report(fair_metrics)
                
                # Store fair results
                experiment_results['model_results'][m]['fair_metrics'] = fair_metrics
                experiment_results['model_results'][m]['improvement_metrics'] = {
                    'accuracy_difference': metrics_base['overall_accuracy'] - fair_metrics['overall_accuracy'],
                    'dp_improvement': (abs(metrics_base.get('demographic_parity_difference', 0)) - 
                                     abs(fair_metrics.get('demographic_parity_difference', 0))),
                    'tpr_improvement': (abs(metrics_base.get('tpr_difference', 0)) - 
                                      abs(fair_metrics.get('tpr_difference', 0)))
                }

        # Save results if folder provided
        if results_folder:
            save_results_to_json(experiment_results, results_folder, 'standard_ml', config['dataname'])

    except Exception as e:
        print(f"An error occurred: {e}")
        experiment_results['error'] = str(e)
        
        # Save error results if folder provided
        if results_folder:
            save_results_to_json(experiment_results, results_folder, 'standard_ml_error', config['dataname'])
    
    return experiment_results

def main(config_path):
    """Standard ML pipeline from the original code."""
    return run_standard_ml_experiment(config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML pipeline with balanced dataset augmentation.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    
    # Standard ML arguments
    parser.add_argument("--fairness", action='store_true', help="Whether to run fairness-aware training.")
    parser.add_argument("--fair_technique", type=str, default=None, help="Fairness technique to use")
    
    # Balanced augmentation arguments
    parser.add_argument("--balanced", action='store_true', 
                       help="Run balanced dataset augmentation experiment")
    parser.add_argument("--api_key", type=str, default="dummy", 
                       help="API key for LLM generation (required for llm_async method)")
    parser.add_argument("--sensitive_ratio", type=float, required=False,
                       help="Target ratio for privileged group (0.0-1.0). E.g., 0.5 = 50:50")
    parser.add_argument("--label_ratio", type=float, required=False,
                       help="Target ratio for positive labels (0.0-1.0). E.g., 0.5 = 50:50")
    parser.add_argument("--scenario_name", type=str, default="balanced",
                       help="Name for the balanced experiment scenario")
    parser.add_argument("--method", type=str, choices=['faker', 'llm_async', 'ctgan', 'tvae'], default='faker',
                       help="Synthetic data generation method")
    
    # Results saving arguments
    parser.add_argument("--save_results", action='store_true', 
                       help="Save results to JSON files in a dedicated folder")
    parser.add_argument("--results_folder", type=str, default=None,
                       help="Custom results folder path (if not specified, auto-generated)")
    
    args = parser.parse_args()

    # Validate method availability and requirements
    if args.method == 'llm_async':
        if not LLM_AVAILABLE:
            print("Error: LLM generator not available")
            print("Please check src/llm_synthetic_generator.py")
            exit(1)
        if not args.api_key or args.api_key == 'dummy':
            print("Error: --method llm_async requires a valid --api_key")
            print("Please provide your DeepSeek API key with --api_key YOUR_KEY")
            exit(1)
    
    if args.method in ['ctgan', 'tvae']:
        if not CTGAN_AVAILABLE:
            print(f"Error: {args.method.upper()} not available. Install with: pip install sdv")
            exit(1)

    # Create results folder if saving is enabled
    results_folder = None
    if args.save_results:
        if args.results_folder:
            results_folder = args.results_folder
            os.makedirs(results_folder, exist_ok=True)
        else:
            results_folder = create_results_folder()
        print(f"Results will be saved to: {results_folder}")

    # Load and update config
    config = load_config(args.config_path)
    if args.fairness:
        config['fairness'] = True
    if args.fair_technique is not None:
        config['fair_technique'] = args.fair_technique
    
    if args.balanced:
        # Check for required ratios
        if args.sensitive_ratio is None or args.label_ratio is None:
            print("Error: For balanced augmentation, both --sensitive_ratio and --label_ratio must be specified")
            print("Example: --sensitive_ratio 0.5 --label_ratio 0.5")
            exit(1)
        
        if not (0.0 <= args.sensitive_ratio <= 1.0) or not (0.0 <= args.label_ratio <= 1.0):
            print("Error: Ratios must be between 0.0 and 1.0")
            exit(1)
        
        print(f"Running balanced augmentation experiment with method: {args.method}")
        results = run_balanced_experiment(
            config_path=args.config_path,
            api_key=args.api_key,
            target_sensitive_ratio=args.sensitive_ratio,
            target_label_ratio=args.label_ratio,
            scenario_name=args.scenario_name,
            results_folder=results_folder,
            method=args.method
        )
        
        if 'error' not in results:
            print(f"\n{'='*80}")
            print("BALANCED AUGMENTATION EXPERIMENT RESULTS")
            print(f"{'='*80}")
            orig = results['original_analysis']
            final = results['final_analysis']
            
            print(f"Generation Method: {results.get('generation_method', 'unknown')}")
            print(f"Original dataset: {orig['total_samples']} samples")
            print(f"Augmented dataset: {final['total_samples']} samples")
            print(f"Samples added: {final['total_samples'] - orig['total_samples']}")
            
            print(f"\nBalance Achievement:")
            print(f"  Sensitive ratio: {orig['current_sensitive_ratio']:.3f} → {final['current_sensitive_ratio']:.3f}")
            print(f"  Label ratio: {orig['current_label_ratio']:.3f} → {final['current_label_ratio']:.3f}")
            
            if results.get('improvement_metrics'):
                print(f"\nFairness Impact:")
                print(f"  Accuracy cost: {results['improvement_metrics'].get('accuracy_difference', 0):.4f}")
                print(f"  DP improvement: {results['improvement_metrics'].get('dp_improvement', 0):.4f}")
        else:
            print(f"Error in experiment: {results['error']}")
    else:
        # Run standard pipeline
        results = run_standard_ml_experiment(args.config_path, results_folder)
        
        print(f"\n{'='*80}")
        print("STANDARD ML EXPERIMENT COMPLETED")
        print(f"{'='*80}")
        
    if results_folder:
        print(f"\n✓ All results saved to: {results_folder}")