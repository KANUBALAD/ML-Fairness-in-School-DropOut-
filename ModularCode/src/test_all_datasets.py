#!/usr/bin/env python3

import pandas as pd
from dynamic_schema_extractor import DynamicSchemaExtractor
import yaml
from pathlib import Path
import os

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

def test_dataset(dataset_name):
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # Go up one level from src/ to ModularCode/
    
    yaml_path = project_root / f"yaml/{dataset_name}.yaml"
    
    print(f"\n=== Testing {dataset_name} ===")
    print(f"Looking for YAML at: {yaml_path}")
    
    if not yaml_path.exists():
        print(f"❌ YAML file not found: {yaml_path}")
        return
        
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with YAML overrides
    yaml_sensitive = config.get('sensitive_column')
    yaml_label = config.get('label_column')
    
    print(f"YAML sensitive column: {yaml_sensitive}")
    print(f"YAML label column: {yaml_label}")
    
    # Load data (handle relative paths)
    data_path = config['datapath']
    if not Path(data_path).is_absolute():
        data_path = project_root / data_path
    
    print(f"Looking for data at: {data_path}")
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return
        
    data = pd.read_csv(data_path)
    extractor = DynamicSchemaExtractor()
    
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
        
        print(f"✅ Final sensitive column: {col_info['sens_col']}")
        print(f"✅ Final label column: {col_info['label_col']}")
        print(f"✅ Sensitive values: {col_info['sens_vals']}")
        print(f"✅ Label values: {col_info['label_vals']}")
        
        # Show sample distribution
        print(f"Sample distribution:")
        sens_dist = dict(data[col_info['sens_col']].value_counts())
        label_dist = dict(data[col_info['label_col']].value_counts())
        print(f"  {col_info['sens_col']}: {sens_dist}")
        print(f"  {col_info['label_col']}: {label_dist}")
        
        # Verify this is what we expect
        expected_sensitive = {'africa': 'gender', 'brazil': 'Gender', 'india': 'STUDENTGENDER'}
        expected_label = {'africa': 'dropout', 'brazil': 'Target', 'india': 'STUDENT_DROPOUT_STATUS'}
        
        if col_info['sens_col'] == expected_sensitive.get(dataset_name):
            print(f"✅ Sensitive column matches expectation!")
        else:
            print(f"⚠️  Expected sensitive: {expected_sensitive.get(dataset_name)}, got: {col_info['sens_col']}")
            
        if col_info['label_col'] == expected_label.get(dataset_name):
            print(f"✅ Label column matches expectation!")
        else:
            print(f"⚠️  Expected label: {expected_label.get(dataset_name)}, got: {col_info['label_col']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Change to the script's directory to ensure relative paths work
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)  # Change to ModularCode/ directory
    
    print(f"Working directory: {os.getcwd()}")
    
    for dataset in ['africa', 'brazil', 'india']:
        try:
            test_dataset(dataset)
        except Exception as e:
            print(f"❌ Error testing {dataset}: {e}")