#!/usr/bin/env python3

import pandas as pd
from dynamic_schema_extractor import DynamicSchemaExtractor
import yaml
from pathlib import Path

def detect_columns(df: pd.DataFrame,
                   extractor: DynamicSchemaExtractor,
                   override_sensitive: str = None,
                   override_label: str = None):
    """Copy of detect_columns function from CTGAN_synthetic_generator.py"""
    schema = extractor.extract_comprehensive_schema(df)
    sens_col = override_sensitive or (
        schema["relationships"]["sensitive_column_candidates"][0]
        if schema["relationships"]["sensitive_column_candidates"] else None)
    label_col = override_label or (
        schema["relationships"]["label_column_candidates"][0]
        if schema["relationships"]["label_column_candidates"] else None)

    if sens_col is None or label_col is None:
        raise ValueError("Could not automatically detect sensitive / label "
                         "columns.  Please supply --sensitive_col / --label_col")

    # Pick the most frequent value as "privileged / negative"
    sens_counts  = df[sens_col].value_counts()
    label_counts = df[label_col].value_counts()

    sens_vals  = tuple(sens_counts.index[:2])   # (priv, unpriv)
    label_vals = tuple(label_counts.index[:2])  # (positive, negative)

    return dict(sens_col=sens_col, label_col=label_col,
                sens_vals=sens_vals, label_vals=label_vals)

def test_dataset(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n=== Testing {config['dataname']} ===")
    
    data = pd.read_csv(config['datapath'])
    extractor = DynamicSchemaExtractor()
    
    # Test with YAML overrides
    yaml_sensitive = config.get('sensitive_column')
    yaml_label = config.get('label_column')
    
    print(f"YAML sensitive column: {yaml_sensitive}")
    print(f"YAML label column: {yaml_label}")
    
    # Extract schema
    schema = extractor.extract_comprehensive_schema(data)
    
    print("Auto-detected sensitive candidates:", 
          schema["relationships"]["sensitive_column_candidates"])
    print("Auto-detected label candidates:", 
          schema["relationships"]["label_column_candidates"])
    
    # Test final column selection (simulate CTGAN logic)
    try:
        col_info = detect_columns(data, extractor, 
                                 override_sensitive=yaml_sensitive,
                                 override_label=yaml_label)
        
        print(f"✓ Final sensitive column: {col_info['sens_col']}")
        print(f"✓ Final label column: {col_info['label_col']}")
        print(f"✓ Sensitive values: {col_info['sens_vals']}")
        print(f"✓ Label values: {col_info['label_vals']}")
        
        # Show sample distribution
        print(f"Sample distribution:")
        print(f"  {col_info['sens_col']}: {dict(data[col_info['sens_col']].value_counts())}")
        print(f"  {col_info['label_col']}: {dict(data[col_info['label_col']].value_counts())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    for dataset in ['africa', 'brazil', 'india']:
        try:
            test_dataset(f'yaml/{dataset}.yaml')
        except FileNotFoundError as e:
            print(f"❌ Could not find YAML or data file for {dataset}: {e}")
        except Exception as e:
            print(f"❌ Error testing {dataset}: {e}")