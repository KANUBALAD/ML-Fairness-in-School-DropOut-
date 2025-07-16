# main.py (v8 - Multi-Provider LLM Integration)

import os
import yaml
import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime

# --- Local Imports ---
from src import utils, dataload, model
import src.fairness as fairness
from src.synthetic_generator import SyntheticDataGenerator
from src.data_quality_analyzer import DataQualityAnalyzer

# Import the new multi-provider LLM generator
try:
    import requests
    from tqdm import tqdm
    from src.llm_synthetic_generator import ThreadedLLMSyntheticGenerator
    LLM_AVAILABLE = True
    print("✓ Multi-provider LLM generator loaded successfully.")
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM generator not available. `requests` or `tqdm` module might be missing.")

# --- Utility Functions ---

def create_results_folder():
    """Create a dedicated results folder with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = f"./results/experiment_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    return results_folder

def save_augmented_dataset_csv(augmented_data, results_folder, config_name, scenario_name, method):
    """Save the augmented dataset as CSV file with enhanced error handling."""
    if results_folder is None:
        print("Warning: No results folder specified. Skipping CSV save.")
        return None
    
    # Create CSV filename with descriptive naming
    csv_filename = f"augmented_{config_name}_{scenario_name}_{method}.csv"
    csv_filepath = os.path.join(results_folder, csv_filename)
    
    try:
        # Ensure augmented_data is a DataFrame
        if not isinstance(augmented_data, pd.DataFrame):
            print(f"❌ Error: augmented_data is not a DataFrame, it's a {type(augmented_data)}")
            return None
        
        # Save the augmented dataset
        augmented_data.to_csv(csv_filepath, index=False)
        print(f"✓ Saved augmented dataset to: {csv_filepath}")
        
        # Also save dataset summary
        summary_filename = f"dataset_summary_{config_name}_{scenario_name}_{method}.txt"
        summary_filepath = os.path.join(results_folder, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            f.write(f"Augmented Dataset Summary\n")
            f.write(f"========================\n")
            f.write(f"Dataset: {config_name}\n")
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(f"Dataset Shape: {augmented_data.shape}\n")
            f.write(f"Total Samples: {len(augmented_data)}\n\n")
            f.write("Column Summary:\n")
            f.write(f"{augmented_data.dtypes}\n\n")
            f.write("First 5 rows:\n")
            f.write(f"{augmented_data.head().to_string()}\n\n")
            f.write("Dataset Info:\n")
            
            # Capture info() output safely
            import io
            buffer = io.StringIO()
            augmented_data.info(buf=buffer)
            f.write(buffer.getvalue())
        
        print(f"✓ Saved dataset summary to: {summary_filepath}")
        return csv_filepath
        
    except Exception as e:
        print(f"❌ Error saving augmented dataset: {e}")
        print(f"   Dataset type: {type(augmented_data)}")
        print(f"   Dataset shape: {getattr(augmented_data, 'shape', 'No shape attribute')}")
        return None
    
def save_generation_log(original_analysis, final_analysis, augmentation_plan, results_folder, config_name, scenario_name, method):
    """Save detailed generation log showing what was augmented."""
    if results_folder is None:
        return
    
    log_filename = f"generation_log_{config_name}_{scenario_name}_{method}.txt"
    log_filepath = os.path.join(results_folder, log_filename)
    
    try:
        with open(log_filepath, 'w') as f:
            f.write(f"Synthetic Data Generation Log\n")
            f.write(f"============================\n")
            f.write(f"Dataset: {config_name}\n")
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Method: {method}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write("ORIGINAL DATASET ANALYSIS:\n")
            f.write(f"Total samples: {original_analysis['total_samples']}\n")
            f.write(f"Privileged samples: {original_analysis['privileged_count']}\n")
            f.write(f"Unprivileged samples: {original_analysis['unprivileged_count']}\n")
            f.write(f"Positive samples: {original_analysis['positive_count']}\n")
            f.write(f"Negative samples: {original_analysis['negative_count']}\n")
            f.write(f"Sensitive ratio: {original_analysis['current_sensitive_ratio']:.4f}\n")
            f.write(f"Label ratio: {original_analysis['current_label_ratio']:.4f}\n\n")
            
            f.write("CROSS-TABULATION (ORIGINAL):\n")
            f.write(f"Privileged + Positive: {original_analysis['cross_tab']['priv_pos']}\n")
            f.write(f"Privileged + Negative: {original_analysis['cross_tab']['priv_neg']}\n")
            f.write(f"Unprivileged + Positive: {original_analysis['cross_tab']['unpriv_pos']}\n")
            f.write(f"Unprivileged + Negative: {original_analysis['cross_tab']['unpriv_neg']}\n\n")
            
            f.write("AUGMENTATION PLAN:\n")
            f.write(f"Total additional samples needed: {augmentation_plan['total_additional']}\n")
            f.write("Breakdown by category:\n")
            for category, count in augmentation_plan['breakdown'].items():
                f.write(f"  {category}: {count} samples\n")
            f.write("\n")
            
            f.write("FINAL DATASET ANALYSIS:\n")
            f.write(f"Total samples: {final_analysis['total_samples']}\n")
            f.write(f"Privileged samples: {final_analysis['privileged_count']}\n")
            f.write(f"Unprivileged samples: {final_analysis['unprivileged_count']}\n")
            f.write(f"Positive samples: {final_analysis['positive_count']}\n")
            f.write(f"Negative samples: {final_analysis['negative_count']}\n")
            f.write(f"Sensitive ratio: {final_analysis['current_sensitive_ratio']:.4f}\n")
            f.write(f"Label ratio: {final_analysis['current_label_ratio']:.4f}\n\n")
            
            f.write("CROSS-TABULATION (FINAL):\n")
            f.write(f"Privileged + Positive: {final_analysis['cross_tab']['priv_pos']}\n")
            f.write(f"Privileged + Negative: {final_analysis['cross_tab']['priv_neg']}\n")
            f.write(f"Unprivileged + Positive: {final_analysis['cross_tab']['unpriv_pos']}\n")
            f.write(f"Unprivileged + Negative: {final_analysis['cross_tab']['unpriv_neg']}\n\n")
            
            samples_added = final_analysis['total_samples'] - original_analysis['total_samples']
            f.write(f"SUMMARY:\n")
            f.write(f"Samples added: {samples_added}\n")
            f.write(f"Original size: {original_analysis['total_samples']}\n")
            f.write(f"Final size: {final_analysis['total_samples']}\n")
            f.write(f"Growth factor: {final_analysis['total_samples'] / original_analysis['total_samples']:.2f}x\n")
        
        print(f"✓ Saved generation log to: {log_filepath}")
        
    except Exception as e:
        print(f"❌ Error saving generation log: {e}")

def save_results_to_json(results, results_folder, experiment_type, config_name, scenario_name=None):
    """Save experiment results to a JSON file."""
    if results_folder is None:
        return
    
    filename = f"{experiment_type}_{config_name}_{scenario_name}.json" if scenario_name else f"{experiment_type}_{config_name}.json"
    filepath = os.path.join(results_folder, filename)
    
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
        return obj

    results_with_metadata = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': experiment_type,
            'config_name': config_name,
            'scenario_name': scenario_name
        },
        'results': convert_numpy_types(results)
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        print(f"✓ Saved results to: {filepath}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def analyze_dataset_imbalance_universal(data, sensitive_col, label_col, privileged_val, positive_label):
    """Universal dataset imbalance analysis that works with any column names and values."""
    
    total_samples = len(data)
    privileged_count = sum(data[sensitive_col] == privileged_val)
    positive_count = sum(data[label_col].astype(str) == str(positive_label))
    
    analysis = {
        'total_samples': total_samples,
        'privileged_count': privileged_count,
        'unprivileged_count': total_samples - privileged_count,
        'current_sensitive_ratio': privileged_count / total_samples if total_samples > 0 else 0,
        'positive_count': positive_count,
        'negative_count': total_samples - positive_count,
        'current_label_ratio': positive_count / total_samples if total_samples > 0 else 0,
        'cross_tab': {
            'priv_pos': sum((data[sensitive_col] == privileged_val) & (data[label_col].astype(str) == str(positive_label))),
            'priv_neg': sum((data[sensitive_col] == privileged_val) & (data[label_col].astype(str) != str(positive_label))),
            'unpriv_pos': sum((data[sensitive_col] != privileged_val) & (data[label_col].astype(str) == str(positive_label))),
            'unpriv_neg': sum((data[sensitive_col] != privileged_val) & (data[label_col].astype(str) != str(positive_label))),
        }
    }
    return analysis

def auto_detect_key_columns(data, config_name):
    """Automatically detect sensitive and label columns using heuristics."""
    
    print(f"🔍 Auto-detecting key columns for dataset: {config_name}")
    
    # Simple heuristics for common column patterns
    sensitive_candidates = []
    label_candidates = []
    
    for col in data.columns:
        col_lower = col.lower()
        
        # Look for sensitive attribute patterns
        if any(keyword in col_lower for keyword in ['gender', 'sex', 'race', 'ethnicity']):
            unique_vals = data[col].nunique()
            if 2 <= unique_vals <= 5:  # Reasonable number of categories
                sensitive_candidates.append(col)
        
        # Look for label patterns
        if any(keyword in col_lower for keyword in ['target', 'label', 'dropout', 'outcome', 'class', 'result']):
            unique_vals = data[col].nunique()
            if 2 <= unique_vals <= 10:  # Reasonable number of outcome categories
                label_candidates.append(col)
    
    # Score candidates based on additional criteria
    def score_sensitive_candidate(col):
        score = 0
        unique_vals = data[col].unique()
        
        # Prefer binary columns
        if len(unique_vals) == 2:
            score += 10
        
        # Check for gender-like values
        str_vals = [str(v).lower() for v in unique_vals]
        gender_indicators = ['male', 'female', 'm', 'f', '0', '1']
        if any(indicator in str_vals for indicator in gender_indicators):
            score += 15
        
        # Prefer balanced distributions
        value_counts = data[col].value_counts()
        if len(value_counts) >= 2:
            balance_ratio = min(value_counts.values) / max(value_counts.values)
            score += balance_ratio * 5
        
        return score
    
    def score_label_candidate(col):
        score = 0
        unique_vals = data[col].unique()
        
        # Prefer binary outcomes
        if len(unique_vals) == 2:
            score += 10
        
        # Check for outcome-like values
        str_vals = [str(v).lower() for v in unique_vals]
        outcome_indicators = ['dropout', 'graduate', 'yes', 'no', 'success', 'fail', '0', '1']
        if any(indicator in str_vals for indicator in outcome_indicators):
            score += 15
        
        # Prefer somewhat imbalanced distributions (typical for outcomes)
        value_counts = data[col].value_counts()
        if len(value_counts) >= 2:
            imbalance_ratio = min(value_counts.values) / max(value_counts.values)
            if 0.1 <= imbalance_ratio <= 0.8:  # Some imbalance is good for outcome variables
                score += 10
        
        return score
    
    # Select best candidates
    best_sensitive = None
    best_label = None
    
    if sensitive_candidates:
        scored_sensitive = [(col, score_sensitive_candidate(col)) for col in sensitive_candidates]
        best_sensitive = max(scored_sensitive, key=lambda x: x[1])[0]
    
    if label_candidates:
        scored_label = [(col, score_label_candidate(col)) for col in label_candidates]
        best_label = max(scored_label, key=lambda x: x[1])[0]
    
    if not best_sensitive or not best_label:
        available_cols = list(data.columns)
        raise ValueError(f"Could not auto-detect key columns. Available columns: {available_cols}")
    
    print(f"✓ Auto-detected: sensitive='{best_sensitive}', label='{best_label}'")
    return best_sensitive, best_label

def determine_privileged_and_positive_values(data, sensitive_col, label_col):
    """Automatically determine privileged and positive values using heuristics."""
    
    # Get unique values
    sensitive_vals = data[sensitive_col].unique().tolist()
    label_vals = data[label_col].unique().tolist()
    
    # Determine privileged value for sensitive attribute
    privileged_val = sensitive_vals[0]  # Default
    if len(sensitive_vals) == 2:
        val1_str, val2_str = str(sensitive_vals[0]).lower(), str(sensitive_vals[1]).lower()
        
        # Gender heuristics
        if 'gender' in sensitive_col.lower() or 'sex' in sensitive_col.lower():
            male_indicators = ['male', 'm', '1', 'man']
            if any(indicator in val1_str for indicator in male_indicators):
                privileged_val = sensitive_vals[0]
            elif any(indicator in val2_str for indicator in male_indicators):
                privileged_val = sensitive_vals[1]
        
        # Numeric heuristics (higher value is privileged)
        try:
            num_vals = [float(v) for v in sensitive_vals]
            privileged_val = max(num_vals)
        except:
            pass
    
    # Determine positive value for label
    positive_val = label_vals[0]  # Default
    if len(label_vals) == 2:
        val1_str, val2_str = str(label_vals[0]).lower(), str(label_vals[1]).lower()
        
        # Outcome heuristics (negative outcomes are typically what we predict)
        negative_indicators = ['dropout', 'fail', 'negative', 'bad', '1']
        if any(indicator in val1_str for indicator in negative_indicators):
            positive_val = label_vals[0]
        elif any(indicator in val2_str for indicator in negative_indicators):
            positive_val = label_vals[1]
    
    print(f"✓ Determined values: privileged='{privileged_val}', positive='{positive_val}'")
    return privileged_val, positive_val

def calculate_augmentation_needs(analysis, target_sensitive_ratio, target_label_ratio):
    """Calculate how many additional samples of each type are needed."""
    current_total = analysis['total_samples']
    
    min_total_sens = 2 * max(analysis['privileged_count'], analysis['unprivileged_count'])
    min_total_label = 2 * max(analysis['positive_count'], analysis['negative_count'])
    target_total = int(np.ceil(max(current_total, min_total_sens, min_total_label)))
    
    target_priv = int(target_total * target_sensitive_ratio)
    target_pos = int(target_total * target_label_ratio)
    
    target_priv_pos = int(target_priv * target_label_ratio)
    target_priv_neg = target_priv - target_priv_pos
    target_unpriv_pos = target_pos - target_priv_pos
    target_unpriv_neg = target_total - target_priv - target_unpriv_pos
    
    current = analysis['cross_tab']
    additional_needed = {
        'total_additional': target_total - current_total,
        'breakdown': {
            'priv_pos': max(0, target_priv_pos - current['priv_pos']),
            'priv_neg': max(0, target_priv_neg - current['priv_neg']),
            'unpriv_pos': max(0, target_unpriv_pos - current['unpriv_pos']),
            'unpriv_neg': max(0, target_unpriv_neg - current['unpriv_neg']),
        }
    }
    return additional_needed

def align_data_types_universal(synthetic_df: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
    """Align synthetic data types with original data types using universal approach."""
    
    print("🔧 Aligning data types universally...")
    aligned_df = synthetic_df.copy()
    
    for col in original_data.columns:
        if col not in aligned_df.columns:
            continue
            
        original_dtype = original_data[col].dtype
        
        try:
            if pd.api.types.is_integer_dtype(original_dtype):
                # Convert to numeric first, then to int
                aligned_df[col] = pd.to_numeric(aligned_df[col], errors='coerce')
                aligned_df[col] = aligned_df[col].fillna(original_data[col].median()).round().astype(original_dtype)
                
            elif pd.api.types.is_float_dtype(original_dtype):
                aligned_df[col] = pd.to_numeric(aligned_df[col], errors='coerce')
                aligned_df[col] = aligned_df[col].fillna(original_data[col].median()).astype(original_dtype)
                
            elif pd.api.types.is_object_dtype(original_dtype):
                aligned_df[col] = aligned_df[col].astype(str)
                # Replace 'nan' strings with actual NaN, then fill
                aligned_df[col] = aligned_df[col].replace('nan', np.nan)
                most_common = original_data[col].mode().iloc[0] if len(original_data[col].mode()) > 0 else 'Unknown'
                aligned_df[col] = aligned_df[col].fillna(most_common)
                
        except Exception as e:
            print(f"   ⚠️  Could not align dtype for column '{col}': {e}")
            continue
    
    return aligned_df

def apply_universal_post_processing(original_data: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Apply universal post-processing enhancements."""
    
    print("✨ Applying universal post-processing...")
    enhanced_df = synthetic_df.copy()
    
    # Apply bounds based on original data for numeric columns
    for col in enhanced_df.columns:
        if col not in original_data.columns:
            continue
            
        if pd.api.types.is_numeric_dtype(enhanced_df[col]) and pd.api.types.is_numeric_dtype(original_data[col]):
            orig_min = original_data[col].min()
            orig_max = original_data[col].max()
            
            # Clip to bounds with small tolerance
            tolerance = (orig_max - orig_min) * 0.05 if orig_max != orig_min else 1
            enhanced_df[col] = enhanced_df[col].clip(orig_min - tolerance, orig_max + tolerance)
            
            # Round integers
            if pd.api.types.is_integer_dtype(original_data[col]):
                enhanced_df[col] = enhanced_df[col].round()
    
    return enhanced_df

def generate_targeted_synthetic_samples(original_data, config_name, api_key, augmentation_plan, method, sensitive_col, label_col, provider=None, model_name=None, base_url=None):
    """Universal synthetic sample generation with multi-provider support."""
    print(f"\nGenerating targeted synthetic samples using method: {method.upper()}")

    # Determine provider and model based on method
    if method == 'llm_deepseek':
        provider = "deepseek"
        model_name = model_name or "deepseek-chat"
        
    elif method == 'llm_huggingface':
        provider = "huggingface" 
        model_name = model_name or "microsoft/DialoGPT-medium"
        
    elif method == 'llm_huggingface_local':
        provider = "huggingface_transformers"
        model_name = model_name or "microsoft/DialoGPT-medium"
        
    elif method == 'llm_openai':
        provider = "openai"
        model_name = model_name or "gpt-3.5-turbo"
        
    elif method == 'llm_custom':
        provider = "custom_openai"
        if not base_url:
            raise ValueError("--base_url is required for llm_custom method")
        model_name = model_name or "default"
        
    elif method == 'llm_async':
        # Legacy support - default to deepseek
        provider = "deepseek"
        model_name = model_name or "deepseek-chat"
        
    elif method == 'faker':
        # Existing Faker implementation
        print("🎲 Using enhanced Faker generation")
        try:
            generator = SyntheticDataGenerator()
            all_synthetic_samples = []
            
            for category_key, needed_count in augmentation_plan['breakdown'].items():
                if needed_count > 0:
                    print(f"  Generating {needed_count} samples for category: {category_key}")
                    category_samples = generator.generate_category_specific_samples(
                        dataset_name=config_name, n_samples=needed_count,
                        category=category_key, reference_data=original_data)
                    all_synthetic_samples.extend(category_samples)
                    print(f"  ✓ Generated {len(category_samples)} samples for {category_key}")
            
            if all_synthetic_samples:
                print(f"🎯 Total synthetic samples generated: {len(all_synthetic_samples)}")
                synthetic_df = pd.DataFrame(all_synthetic_samples)
                
                # Ensure all columns are present and in correct order
                for col in original_data.columns:
                    if col not in synthetic_df.columns:
                        if pd.api.types.is_numeric_dtype(original_data[col]):
                            default_val = original_data[col].median()
                        else:
                            default_val = original_data[col].mode().iloc[0] if len(original_data[col].mode()) > 0 else 'Unknown'
                        synthetic_df[col] = default_val
                
                synthetic_df = synthetic_df[original_data.columns]
                synthetic_df = align_data_types_universal(synthetic_df, original_data)
                
                augmented_data = pd.concat([original_data, synthetic_df], ignore_index=True)
                print(f"📊 Final augmented dataset: {len(original_data)} original + {len(synthetic_df)} synthetic = {len(augmented_data)} total")
                return augmented_data
            else:
                print("❌ Faker generation produced no valid samples. Returning original data.")
                return original_data.copy()
                
        except Exception as e:
            print(f"💥 Error in Faker generation: {e}")
            print("🔄 Returning original data without augmentation.")
            return original_data.copy()
    else:
        raise ValueError(f"Unknown generation method: {method}")

    # LLM generation (any provider)
    if not LLM_AVAILABLE:
        print("Error: LLM generator not available. Falling back to Faker.")
        return generate_targeted_synthetic_samples(original_data, config_name, api_key, augmentation_plan, 'faker', sensitive_col, label_col)
    
    try:
        print(f"🚀 Using {provider.upper()} LLM generation")
        print(f"   Model: {model_name}")
        
        # Determine optimal worker count based on provider
        if provider == "huggingface_transformers":
            max_workers = 1  # Local models should use fewer workers
        else:
            max_workers = 6  # API-based providers can handle more workers
        
        generator = ThreadedLLMSyntheticGenerator(
            api_key=api_key,
            provider=provider,
            model_name=model_name,
            base_url=base_url,
            max_workers=max_workers
        )
        
        breakdown = augmentation_plan['breakdown']
        target_specs = []
        
        for category, count in breakdown.items():
            if count > 0:
                target_specs.append({
                    'category': category, 
                    'count': count
                })
        
        print(f"📋 Generation plan: {[(spec['category'], spec['count']) for spec in target_specs]}")
        
        synthetic_samples = generator.generate_samples(config_name, target_specs, original_data)
        
        if synthetic_samples:
            print(f"✅ Generated {len(synthetic_samples)} synthetic samples")
            
            synthetic_df = pd.DataFrame(synthetic_samples)
            
            # Ensure all columns are present
            for col in original_data.columns:
                if col not in synthetic_df.columns:
                    if pd.api.types.is_numeric_dtype(original_data[col]):
                        default_val = original_data[col].median()
                    else:
                        default_val = original_data[col].mode().iloc[0] if len(original_data[col].mode()) > 0 else 'Unknown'
                    synthetic_df[col] = default_val
            
            synthetic_df = synthetic_df[original_data.columns]
            synthetic_df = align_data_types_universal(synthetic_df, original_data)
            synthetic_df = apply_universal_post_processing(original_data, synthetic_df)
            
            augmented_data = pd.concat([original_data, synthetic_df], ignore_index=True)
            
            print(f"📊 Final augmented dataset: {len(original_data)} original + {len(synthetic_df)} synthetic = {len(augmented_data)} total")
            return augmented_data
        else:
            print("❌ LLM generation produced no valid samples. Returning original data.")
            return original_data.copy()
            
    except Exception as e:
        print(f"💥 Critical error occurred during {provider.upper()} generation: {e}")
        print("🔄 Falling back to Faker method...")
        return generate_targeted_synthetic_samples(original_data, config_name, api_key, augmentation_plan, 'faker', sensitive_col, label_col)

def run_balanced_experiment(config, api_key, target_sensitive_ratio, target_label_ratio, scenario_name, results_folder, method, provider=None, model_name=None, base_url=None):
    """Universal balanced augmentation experiment with enhanced error handling."""
    print(f"\n{'='*80}\nUNIVERSAL BALANCED AUGMENTATION EXPERIMENT: {scenario_name} ({method.upper()})\n{'='*80}")
    
    try:
        # Load original data
        original_data = pd.read_csv(config['datapath'])
        config_name = config['dataname']
        
        # Auto-detect key columns if not specified
        if 'columns' not in config:
            print("🔍 No column mapping found in config. Auto-detecting key columns...")
            sensitive_col, label_col = auto_detect_key_columns(original_data, config_name)
            privileged_val, positive_val = determine_privileged_and_positive_values(original_data, sensitive_col, label_col)
            
            universal_config = create_universal_config(original_data, config_name, sensitive_col, label_col, privileged_val, positive_val)
            config.update(universal_config)
        else:
            sensitive_col = config['columns']['sensitive_col']
            label_col = config['columns']['label_col']
            privileged_val = config['columns']['privileged_val']
            positive_val = config['columns']['positive_label']
        
        print(f"📊 Using columns: sensitive='{sensitive_col}', label='{label_col}'")
        print(f"🎯 Using values: privileged='{privileged_val}', positive='{positive_val}'")
        
        # Analyze current dataset
        analysis = analyze_dataset_imbalance_universal(original_data, sensitive_col, label_col, privileged_val, positive_val)
        
        if target_sensitive_ratio == "original": 
            target_sensitive_ratio = analysis['current_sensitive_ratio']
        if target_label_ratio == "original": 
            target_label_ratio = analysis['current_label_ratio']

        augmentation_plan = calculate_augmentation_needs(analysis, float(target_sensitive_ratio), float(target_label_ratio))
        
        if augmentation_plan['total_additional'] <= 0:
            print("No augmentation needed or target ratios already met.")
            augmented_data = original_data.copy()
        else:
            augmented_data = generate_targeted_synthetic_samples(
                original_data, config_name, api_key, augmentation_plan, method, 
                sensitive_col, label_col, provider, model_name, base_url
            )

        # CRITICAL: Check if augmented_data is None
        if augmented_data is None:
            print("❌ Critical error: Augmented data is None. Using original data.")
            augmented_data = original_data.copy()

        final_analysis = analyze_dataset_imbalance_universal(augmented_data, sensitive_col, label_col, privileged_val, positive_val)
        print(f"\n📊 Final Distribution: Sensitive Ratio={final_analysis['current_sensitive_ratio']:.3f}, Label Ratio={final_analysis['current_label_ratio']:.3f}")
        
        # Save results with error handling
        csv_path = None
        quality_analysis_path = None
        
        if results_folder:
            csv_path = save_augmented_dataset_csv(augmented_data, results_folder, config_name, scenario_name, method)
            save_generation_log(analysis, final_analysis, augmentation_plan, results_folder, config_name, scenario_name, method)
            
            # Data Quality Analysis with error handling
            try:
                print(f"\n{'='*60}\nDATA QUALITY ANALYSIS\n{'='*60}")
                analyzer = DataQualityAnalyzer()
                
                quality_analysis = analyzer.comprehensive_comparison(
                    original_data=original_data,
                    augmented_data=augmented_data,
                    sensitive_col=sensitive_col,
                    label_col=label_col,
                    dataset_name=config_name,
                    scenario_name=scenario_name,
                    generation_method=method
                )
                
                quality_analysis_path = os.path.join(results_folder, f"data_quality_analysis_{config_name}_{scenario_name}_{method}.json")
                analyzer.save_analysis(quality_analysis, quality_analysis_path)
                
                # Print key insights
                gen_scores = quality_analysis.get('generation_quality_scores', {})
                print(f"\n📊 Data Quality Scores:")
                print(f"   Overall Generation Score: {gen_scores.get('overall_generation_score', 0):.3f}")
                print(f"   Fidelity Score: {gen_scores.get('fidelity_score', 0):.3f}")
                print(f"   Diversity Score: {gen_scores.get('diversity_score', 0):.3f}")
                print(f"   Privacy Score: {gen_scores.get('privacy_score', 0):.3f}")
                print(f"   Utility Score: {gen_scores.get('utility_score', 0):.3f}")
                
                recommendations = quality_analysis.get('recommendations', [])
                if recommendations:
                    print(f"\n💡 Recommendations:")
                    for i, rec in enumerate(recommendations[:3], 1):
                        print(f"   {i}. {rec}")
                        
            except Exception as e:
                print(f"❌ Error in data quality analysis: {e}")
                quality_analysis_path = None
        
        # Test fairness
        baseline_metrics, fair_metrics = test_fairness_on_dataset_universal(
            augmented_data, config, sensitive_col, label_col, f"Balanced_{scenario_name}", 
            fairness_on=config.get('fairness', False)
        )
        
        results = {
            'scenario_name': scenario_name, 
            'generation_method': method, 
            'original_analysis': analysis, 
            'final_analysis': final_analysis, 
            'augmentation_plan': augmentation_plan,
            'baseline_metrics': baseline_metrics, 
            'fair_metrics': fair_metrics,
            'csv_path': csv_path,
            'quality_analysis_path': quality_analysis_path,
            'detected_columns': {
                'sensitive_col': sensitive_col,
                'label_col': label_col,
                'privileged_val': privileged_val,
                'positive_val': positive_val
            }
        }
        
        if fair_metrics:
            results['improvement_metrics'] = {
                'accuracy_cost': baseline_metrics['overall_accuracy'] - fair_metrics['overall_accuracy'],
                'dp_improvement': abs(baseline_metrics.get('demographic_parity_difference', 0)) - abs(fair_metrics.get('demographic_parity_difference', 0))
            }
            print(f"\nDP improvement: {results['improvement_metrics']['dp_improvement']:.4f}, Accuracy cost: {results['improvement_metrics']['accuracy_cost']:.4f}")
        
        if results_folder:
            save_results_to_json(results, results_folder, 'balanced_augmentation', config_name, scenario_name)
        
        return results
        
    except Exception as e:
        print(f"💥 Critical error in run_balanced_experiment: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def create_universal_config(data, config_name, sensitive_col, label_col, privileged_val, positive_val):
    """Create a universal config structure for compatibility with existing functions."""
    return {
        'columns': {
            'sensitive_col': sensitive_col,
            'label_col': label_col,
            'privileged_val': privileged_val,
            'positive_label': positive_val
        }
    }

def test_fairness_on_dataset_universal(data, config_template, sensitive_col, label_col, model_prefix, fairness_on=True):
    """Universal fairness testing that works with any dataset."""
    temp_path = f"./data/temp_{model_prefix}.csv"
    data.to_csv(temp_path, index=False)
    
    # Create temporary config for this specific dataset
    temp_config = config_template.copy()
    temp_config['datapath'] = temp_path
    
    try:
        X_transformed, y_target = dataload.load_data(temp_config)
        sensitive_attr = utils.extract_sensitive_attribute(data, temp_config)
        split = utils.split_data(np.array(X_transformed), np.array(y_target), sens=sensitive_attr, 
                                test_size=temp_config.get('test_size', 0.2), 
                                random_state=temp_config.get('random_state', 42))
        X_train, X_test, y_train, y_test, sens_train, sens_test = split['X_train'], split['X_test'], split['y_train'], split['y_test'], split['sens_train'], split['sens_test']
        
        from sklearn.linear_model import LogisticRegression
        baseline_model = LogisticRegression(max_iter=1000, random_state=temp_config.get('random_state', 42))
        baseline_model.fit(X_train, y_train)
        y_pred_baseline = baseline_model.predict(X_test)
        baseline_metrics = utils.fairness_summary(y_pred_baseline, y_test, sens_test, model_name=f"{model_prefix}_Baseline")
        
        fair_metrics = None
        if fairness_on:
            fair_results = fairness.run_fairness_aware_training(
                np.array(X_transformed), np.array(y_target), sensitive_attr, 
                model_type='logistic_regression', 
                technique=temp_config.get('fair_technique', 'reweighting'), 
                test_size=temp_config.get('test_size', 0.2), 
                random_state=temp_config.get('random_state', 42))
            fair_metrics = utils.fairness_summary(fair_results['y_pred_fair'], fair_results['y_test'], fair_results['sens_test'], model_name=f"{model_prefix}_Fair")

        return baseline_metrics, fair_metrics
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def run_standard_ml_experiment(config, results_folder, scenario_name):
    """Universal standard ML experiment."""
    print(f"\n{'='*80}\nUNIVERSAL STANDARD ML EXPERIMENT: {scenario_name}\n{'='*80}")
    
    original_data = pd.read_csv(config['datapath'])
    config_name = config['dataname']
    
    # Auto-detect columns if needed
    if 'columns' not in config:
        sensitive_col, label_col = auto_detect_key_columns(original_data, config_name)
        privileged_val, positive_val = determine_privileged_and_positive_values(original_data, sensitive_col, label_col)
        universal_config = create_universal_config(original_data, config_name, sensitive_col, label_col, privileged_val, positive_val)
        config.update(universal_config)
    else:
        sensitive_col = config['columns']['sensitive_col']
        label_col = config['columns']['label_col']
        privileged_val = config['columns']['privileged_val']
        positive_val = config['columns']['positive_label']
    
    analysis = analyze_dataset_imbalance_universal(original_data, sensitive_col, label_col, privileged_val, positive_val)
    baseline_metrics, fair_metrics = test_fairness_on_dataset_universal(
        original_data, config, sensitive_col, label_col, "Standard", 
        fairness_on=config.get('fairness', False)
    )
    
    # Save results
    original_csv_path = None
    if results_folder:
        original_csv_path = save_augmented_dataset_csv(original_data, results_folder, config_name, f"{scenario_name}_original", "none")
    
    results = {
        'scenario_name': scenario_name, 
        'original_analysis': analysis, 
        'baseline_metrics': baseline_metrics, 
        'fair_metrics': fair_metrics,
        'csv_path': original_csv_path,
        'detected_columns': {
            'sensitive_col': sensitive_col,
            'label_col': label_col,
            'privileged_val': privileged_val,
            'positive_val': positive_val
        }
    }
    
    if fair_metrics:
        results['improvement_metrics'] = {
            'accuracy_cost': baseline_metrics['overall_accuracy'] - fair_metrics['overall_accuracy'],
            'dp_improvement': abs(baseline_metrics.get('demographic_parity_difference', 0)) - abs(fair_metrics.get('demographic_parity_difference', 0))
        }
        print(f"\nDP improvement: {results['improvement_metrics']['dp_improvement']:.4f}, Accuracy cost: {results['improvement_metrics']['accuracy_cost']:.4f}")
    
    if results_folder:
        save_results_to_json(results, results_folder, 'standard_ml', config_name, scenario_name)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal ML pipeline with multi-provider synthetic data generation.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    parser.add_argument("--fairness", action='store_true', help="Enable fairness-aware training.")
    parser.add_argument("--balanced", action='store_true', help="Run balanced dataset augmentation.")
    
    # Enhanced method choices with multi-provider support
    parser.add_argument("--method", type=str, 
                       choices=['faker', 'llm_async', 'llm_deepseek', 'llm_huggingface', 
                               'llm_huggingface_local', 'llm_openai', 'llm_custom'], 
                       default='faker', 
                       help="Synthetic data generation method.")
    
    # Provider-specific arguments
    parser.add_argument("--model_name", type=str, default=None, 
                       help="Specific model name (e.g., 'google/flan-t5-large', 'gpt-4', 'deepseek-chat')")
    parser.add_argument("--base_url", type=str, default=None, 
                       help="Custom API base URL (required for llm_custom)")
    
    parser.add_argument("--sensitive_ratio", type=str, required=False, help="Target ratio for privileged group (e.g., 0.5 or 'original').")
    parser.add_argument("--label_ratio", type=str, required=False, help="Target ratio for positive labels (e.g., 0.5 or 'original').")
    parser.add_argument("--api_key", type=str, default="dummy", help="API key for LLM generation.")
    parser.add_argument("--scenario_name", type=str, default="default_scenario", help="Name for the experiment scenario.")
    parser.add_argument("--save_results", action='store_true', help="Save results to JSON files.")
    parser.add_argument("--results_folder", type=str, default=None, help="Custom results folder path.")
    
    args = parser.parse_args()

    # --- SETUP AND CONFIGURATION ---
    config = load_config(args.config_path)
    config['fairness'] = args.fairness
    
    print(f"🎯 Running UNIVERSAL pipeline with {args.method.upper()} generation!")

    # --- Validation ---
    if args.balanced and (args.sensitive_ratio is None or args.label_ratio is None):
        parser.error("--balanced requires --sensitive_ratio and --label_ratio to be set.")
    
    # Enhanced validation for LLM methods
    llm_methods = ['llm_async', 'llm_deepseek', 'llm_huggingface', 'llm_openai', 'llm_custom']
    if args.method in llm_methods:
        if not LLM_AVAILABLE:
            parser.error(f"--method {args.method} requires necessary libraries (requests, tqdm).")
        if args.method != 'llm_huggingface_local' and args.api_key == 'dummy':
            parser.error(f"--method {args.method} requires a valid --api_key.")
        if args.method == 'llm_custom' and not args.base_url:
            parser.error("--method llm_custom requires --base_url to be set.")

    # Create results folder
    if args.save_results or args.balanced:
        if args.results_folder:
            results_folder = args.results_folder
            os.makedirs(results_folder, exist_ok=True)
            print(f"Created/verified results folder: {results_folder}")
        else:
            results_folder = create_results_folder()
        print(f"Results will be saved to: {results_folder}")
    else:
        results_folder = None

    # --- EXECUTION ---
    if args.balanced:
        run_balanced_experiment(
            config=config, api_key=args.api_key, target_sensitive_ratio=args.sensitive_ratio,
            target_label_ratio=args.label_ratio, scenario_name=args.scenario_name,
            results_folder=results_folder, method=args.method, 
            provider=None, model_name=args.model_name, base_url=args.base_url
        )
    else:
        run_standard_ml_experiment(config=config, results_folder=results_folder, scenario_name=args.scenario_name)

    print(f"\n{'='*80}\nUNIVERSAL EXPERIMENT '{args.scenario_name}' COMPLETED.\n{'='*80}")