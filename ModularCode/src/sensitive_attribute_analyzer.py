#!/usr/bin/env python3
"""
Sensitive Attribute Distribution Analyzer
========================================

Compares the distribution of sensitive attributes between original and synthetic data.
Uses YAML configuration and improved DynamicSchemaExtractor for consistency.
"""

import sys
import yaml
import pandas as pd
from pathlib import Path
from dynamic_schema_extractor import DynamicSchemaExtractor

def load_yaml(path):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def detect_columns(df, extractor, yaml_config):
    """
    Detect sensitive and label columns using the same logic as CTGAN_synthetic_generator.py
    Uses YAML overrides if available, falls back to auto-detection.
    """
    # Get YAML overrides
    yaml_sensitive = yaml_config.get('sensitive_column')
    yaml_label = yaml_config.get('label_column')
    
    # Extract schema
    schema = extractor.extract_comprehensive_schema(df)
    
    # Use YAML override or auto-detection
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
            f"Available label candidates: {available_label}. "
            f"Please specify sensitive_column and label_column in YAML config."
        )
    
    # Get value distributions for understanding privileged/unprivileged groups
    sens_counts = df[sens_col].value_counts()
    label_counts = df[label_col].value_counts()
    
    sens_vals = tuple(sens_counts.index[:2])   # (most frequent, second most frequent)
    label_vals = tuple(label_counts.index[:2]) # (most frequent, second most frequent)
    
    return {
        'sens_col': sens_col,
        'label_col': label_col,
        'sens_vals': sens_vals,
        'label_vals': label_vals,
        'sens_counts': sens_counts,
        'label_counts': label_counts
    }

def determine_privileged_group(sens_counts, sens_col):
    """
    Determine which group should be considered 'privileged' based on dataset context.
    This uses heuristics based on column name and values.
    """
    values = sens_counts.index.tolist()
    col_lower = sens_col.lower()
    
    # Gender-based heuristics
    if 'gender' in col_lower:
        # Check for common gender encodings
        str_values = [str(v).lower() for v in values]
        
        # In many fairness contexts, male is often considered the privileged group
        if 'male' in str_values and 'female' in str_values:
            return 'Male' if 'Male' in values else 'male'
        elif 'f' in str_values and 'm' in str_values:
            return 'M' if 'M' in values else 'm'
        elif 0 in values and 1 in values:
            # Common encoding: 0=Female, 1=Male (but varies by dataset)
            # Check which is more frequent and assume it's the privileged group
            # This is a heuristic and may need dataset-specific adjustment
            return sens_counts.index[0]  # Most frequent group
    
    # For other sensitive attributes, use most frequent as privileged
    # (this is a common assumption but may need adjustment per dataset)
    return sens_counts.index[0]

def analyze_sensitive_distribution(original_df, synthetic_df, col_info):
    """Analyze the distribution of sensitive attributes."""
    sens_col = col_info['sens_col']
    
    # Original distribution
    orig_counts = original_df[sens_col].value_counts()
    orig_total = len(original_df)
    
    # Synthetic distribution  
    synth_counts = synthetic_df[sens_col].value_counts()
    synth_total = len(synthetic_df)
    
    # Determine privileged group
    privileged_val = determine_privileged_group(orig_counts, sens_col)
    
    # Get all unique values and ensure we have at least 2
    all_values = sorted(set(orig_counts.index) | set(synth_counts.index))
    
    if len(all_values) < 2:
        raise ValueError(f"Sensitive attribute {sens_col} must have at least 2 values for analysis")
    
    # Find unprivileged group (the other main group)
    unprivileged_val = None
    for val in all_values:
        if val != privileged_val:
            unprivileged_val = val
            break
    
    # Calculate statistics for both groups
    results = {}
    for group_name, val in [('privileged', privileged_val), ('unprivileged', unprivileged_val)]:
        orig_count = orig_counts.get(val, 0)
        synth_count = synth_counts.get(val, 0)
        
        orig_pct = (orig_count / orig_total) * 100 if orig_total > 0 else 0
        synth_pct = (synth_count / synth_total) * 100 if synth_total > 0 else 0
        
        results[group_name] = {
            'value': val,
            'original_count': orig_count,
            'original_pct': orig_pct,
            'synthetic_count': synth_count,
            'synthetic_pct': synth_pct,
            'difference_pct': synth_pct - orig_pct
        }
    
    return results, sens_col

def main():
    if len(sys.argv) != 5:
        print("Usage: python sensitive_attribute_analyzer.py <original_csv> <synthetic_csv> <yaml_config> <ratio_value>", file=sys.stderr)
        print("", file=sys.stderr)
        print("This script analyzes the distribution of sensitive attributes between original and synthetic data.", file=sys.stderr)
        print("It uses YAML configuration for column detection and outputs CSV-formatted results.", file=sys.stderr)
        sys.exit(1)
    
    original_csv = sys.argv[1]
    synthetic_csv = sys.argv[2]
    yaml_config_path = sys.argv[3]
    ratio_value = float(sys.argv[4])
    
    try:
        # Load configuration
        yaml_config = load_yaml(yaml_config_path)
        
        # Load data
        original_df = pd.read_csv(original_csv)
        synthetic_df = pd.read_csv(synthetic_csv)
        
        # Detect columns using consistent logic
        extractor = DynamicSchemaExtractor()
        col_info = detect_columns(original_df, extractor, yaml_config)
        
        # Verify synthetic data has the same sensitive column
        if col_info['sens_col'] not in synthetic_df.columns:
            raise ValueError(f"Sensitive column '{col_info['sens_col']}' not found in synthetic data")
        
        # Analyze distributions
        results, sens_col = analyze_sensitive_distribution(original_df, synthetic_df, col_info)
        
        # Output CSV row format: ratio,orig_priv,orig_unpriv,orig_priv_pct,synth_priv,synth_unpriv,synth_priv_pct,diff_pct
        priv = results['privileged']
        unpriv = results['unprivileged']
        
        print(f"{ratio_value:.2f},"
              f"{priv['original_count']},"
              f"{unpriv['original_count']},"
              f"{priv['original_pct']:.2f},"
              f"{priv['synthetic_count']},"
              f"{unpriv['synthetic_count']},"
              f"{priv['synthetic_pct']:.2f},"
              f"{priv['difference_pct']:.2f}")
              
        # Optional: Print debug info to stderr (won't interfere with CSV output)
        print(f"DEBUG: Dataset={yaml_config.get('dataname', 'unknown')}, "
              f"SensCol={sens_col}, "
              f"Privileged={priv['value']}, "
              f"Unprivileged={unpriv['value']}", file=sys.stderr)
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found - {e}", file=sys.stderr)
        print(f"{ratio_value:.2f},FILE_ERROR,FILE_ERROR,FILE_ERROR,FILE_ERROR,FILE_ERROR,FILE_ERROR,FILE_ERROR")
        sys.exit(1)
        
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"{ratio_value:.2f},VALUE_ERROR,VALUE_ERROR,VALUE_ERROR,VALUE_ERROR,VALUE_ERROR,VALUE_ERROR,VALUE_ERROR")
        sys.exit(1)
        
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        print(f"{ratio_value:.2f},UNKNOWN_ERROR,UNKNOWN_ERROR,UNKNOWN_ERROR,UNKNOWN_ERROR,UNKNOWN_ERROR,UNKNOWN_ERROR,UNKNOWN_ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()