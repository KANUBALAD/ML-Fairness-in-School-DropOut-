#!/usr/bin/env python3
"""
Universal Joint Fairness Analyzer
---------------------------------
Updated to use YAML configurations and improved DynamicSchemaExtractor for consistency.

Given:
- results_folder: path containing synthetic csv files named like {dataset}_Sens_X_Label_Y.csv
- original_csv: original dataset for reference
- yaml_config: YAML configuration file with sensitive_column and label_column specifications

Outputs a joint distribution parity summary as 'joint_distribution_analysis.csv' in the same folder.

Usage:
    python analyze_fairness_joint.py <results_folder> <original_csv> <yaml_config>
"""

import os
import re
import sys
import csv
import yaml
import pandas as pd
from pathlib import Path

# Add the src directory to Python path for imports
script_dir = Path(__file__).parent
src_dir = script_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Now import our updated schema extractor
try:
    from dynamic_schema_extractor import DynamicSchemaExtractor
except ImportError:
    print("❌ ERROR: Could not import DynamicSchemaExtractor", file=sys.stderr)
    print("   Make sure dynamic_schema_extractor.py is in the src/ directory", file=sys.stderr)
    sys.exit(1)

def load_yaml(path):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def detect_columns(df, extractor, yaml_config):
    """
    Detect sensitive and label columns using the same logic as other scripts.
    Uses YAML overrides if available, falls back to auto-detection.
    """
    # Get YAML overrides
    yaml_sensitive = yaml_config.get('sensitive_column')
    yaml_label = yaml_config.get('label_column')
    
    print(f"  YAML overrides: sensitive='{yaml_sensitive}', label='{yaml_label}'")
    
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
    
    return sens_col, label_col

def determine_privileged_groups(df, sens_col, label_col):
    """
    Determine which values should be considered 'privileged' for reporting consistency.
    """
    sens_counts = df[sens_col].value_counts()
    label_counts = df[label_col].value_counts()
    
    # For sensitive attribute: use same logic as analyzer
    col_lower = sens_col.lower()
    if 'gender' in col_lower:
        str_values = [str(v).lower() for v in sens_counts.index]
        if 'male' in str_values and 'female' in str_values:
            privileged_sens = 'Male' if 'Male' in sens_counts.index else 'male'
        elif 'f' in str_values and 'm' in str_values:
            privileged_sens = 'M' if 'M' in sens_counts.index else 'm'
        else:
            privileged_sens = sens_counts.index[0]  # Most frequent
    else:
        privileged_sens = sens_counts.index[0]  # Most frequent
    
    # For label: typically the negative outcome is more frequent
    # Look for common patterns
    label_values = [str(v).lower() for v in label_counts.index]
    if any(word in ' '.join(label_values) for word in ['dropout', 'no', 'negative']):
        # Find the negative/unfavorable outcome
        for val in label_counts.index:
            val_lower = str(val).lower()
            if any(neg in val_lower for neg in ['no', 'dropout', 'negative', 'fail']):
                privileged_label = val  # Actually unfavorable, but we call it "reference"
                break
        else:
            privileged_label = label_counts.index[0]  # Most frequent
    else:
        privileged_label = label_counts.index[0]  # Most frequent
    
    return privileged_sens, privileged_label

def analyze_joint_distribution(df, sens_col, label_col):
    """Analyze joint distribution of sensitive and label attributes."""
    # Create crosstab
    table = pd.crosstab(df[sens_col], df[label_col], dropna=False)
    total = table.values.sum()
    
    results = {}
    for sens_val in table.index:
        for label_val in table.columns:
            cnt = table.loc[sens_val, label_val]
            pct = (cnt / total) * 100 if total > 0 else 0.0
            results[(sens_val, label_val)] = {
                'count': cnt,
                'percentage': pct
            }
    
    return results, table

def extract_ratios_from_filename(filename: str):
    """
    Extract ratio information from filename.
    Returns (ratio_type, ratio_value) where ratio_type is 'sensitive' or 'label'
    """
    # Match patterns like dataset_Sens_45_Label_50.csv
    match = re.search(r'Sens_(\d+)_Label_(\d+)', filename)
    if not match:
        return ("unknown", None)
    
    sens_val = int(match.group(1))
    label_val = int(match.group(2))
    
    # Determine which dimension is being varied
    if label_val == 50 and sens_val != 50:
        # Label is fixed at 50%, sensitive is being varied
        return ("sensitive", sens_val / 100.0)  # Convert to decimal
    elif sens_val == 50 and label_val != 50:
        # Sensitive is fixed at 50%, label is being varied
        return ("label", label_val / 100.0)  # Convert to decimal
    elif sens_val == 50 and label_val == 50:
        # Both at 50% - this is the baseline
        return ("baseline", 0.50)
    else:
        # Both being varied or unclear
        return ("both", None)

def main():
    if len(sys.argv) != 4:
        print("Usage: python analyze_fairness_joint.py <results_folder> <original_csv> <yaml_config>", file=sys.stderr)
        print("", file=sys.stderr)
        print("This script analyzes joint distribution of sensitive and label attributes", file=sys.stderr)
        print("between original and synthetic datasets.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Arguments:", file=sys.stderr)
        print("  results_folder: Directory containing synthetic CSV files", file=sys.stderr)
        print("  original_csv:   Path to original dataset CSV", file=sys.stderr)
        print("  yaml_config:    YAML configuration file with column specifications", file=sys.stderr)
        sys.exit(1)

    results_folder, original_csv, yaml_config_path = sys.argv[1:]
    
    try:
        # Load configuration
        yaml_config = load_yaml(yaml_config_path)
        dataset_name = yaml_config.get('dataname', 'unknown')
        
        print(f"Processing dataset: {dataset_name}")
        print(f"Results folder: {results_folder}")
        print(f"Original data: {original_csv}")
        print(f"YAML config: {yaml_config_path}")
        
        # Load original dataset and detect columns
        original_df = pd.read_csv(original_csv)
        extractor = DynamicSchemaExtractor()
        
        sens_col, label_col = detect_columns(original_df, extractor, yaml_config)
        print(f"✅ Detected columns: Sensitive='{sens_col}', Label='{label_col}'")
        
        # Determine privileged groups for consistent reporting
        privileged_sens, privileged_label = determine_privileged_groups(original_df, sens_col, label_col)
        print(f"✅ Reference groups: Sensitive='{privileged_sens}', Label='{privileged_label}'")
        
        # Analyze original dataset
        orig_results, orig_table = analyze_joint_distribution(original_df, sens_col, label_col)
        print(f"✅ Original joint distribution:")
        print(orig_table)
        print()
        
        # Find all synthetic CSV files
        if not os.path.exists(results_folder):
            raise FileNotFoundError(f"Results folder not found: {results_folder}")
        
        csv_files = [f for f in os.listdir(results_folder) 
                    if f.endswith('.csv') and 'analysis' not in f.lower()]
        csv_files.sort()
        
        if not csv_files:
            raise ValueError(f"No synthetic CSV files found in {results_folder}")
        
        print(f"Found {len(csv_files)} synthetic files to analyze")
        
        # Prepare output file
        output_file = os.path.join(results_folder, "joint_distribution_analysis.csv")
        
        # CSV header
        header = [
            "synthetic_filename", "ratio_type", "ratio_value",
            "sensitive_value", "label_value", 
            "orig_count", "orig_pct", "synth_count", "synth_pct", "diff_pct",
            "group_name"  # Add friendly group name
        ]
        
        processed_count = 0
        error_count = 0
        
        with open(output_file, "w", newline="", encoding="utf-8") as out_fh:
            writer = csv.writer(out_fh)
            writer.writerow(header)
            
            for filename in csv_files:
                synthetic_path = os.path.join(results_folder, filename)
                
                try:
                    print(f"  Processing: {filename}")
                    
                    # Load synthetic data
                    synthetic_df = pd.read_csv(synthetic_path)
                    
                    # Verify columns exist
                    if sens_col not in synthetic_df.columns or label_col not in synthetic_df.columns:
                        print(f"    ⚠️  Missing columns in {filename}")
                        error_count += 1
                        continue
                    
                    # Analyze synthetic distribution
                    synth_results, synth_table = analyze_joint_distribution(synthetic_df, sens_col, label_col)
                    
                    # Extract ratio information
                    ratio_type, ratio_value = extract_ratios_from_filename(filename)
                    
                    # Get all possible (sensitive, label) combinations
                    all_combinations = set(orig_results.keys()) | set(synth_results.keys())
                    all_combinations = sorted(all_combinations, key=lambda x: (str(x[0]), str(x[1])))
                    
                    # Write results for each combination
                    for sens_val, label_val in all_combinations:
                        # Original data
                        orig_data = orig_results.get((sens_val, label_val), {'count': 0, 'percentage': 0.0})
                        orig_count = orig_data['count']
                        orig_pct = orig_data['percentage']
                        
                        # Synthetic data
                        synth_data = synth_results.get((sens_val, label_val), {'count': 0, 'percentage': 0.0})
                        synth_count = synth_data['count']
                        synth_pct = synth_data['percentage']
                        
                        # Calculate difference
                        diff_pct = synth_pct - orig_pct
                        
                        # Create friendly group name
                        group_name = f"{sens_val} & {label_val}"
                        
                        # Write CSV row
                        writer.writerow([
                            filename,
                            ratio_type,
                            f"{ratio_value:.2f}" if ratio_value is not None else "",
                            sens_val,
                            label_val,
                            orig_count,
                            f"{orig_pct:.2f}",
                            synth_count,
                            f"{synth_pct:.2f}",
                            f"{diff_pct:.2f}",
                            group_name
                        ])
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"    ❌ Error processing {filename}: {e}")
                    error_count += 1
                    
                    # Write error row
                    writer.writerow([
                        filename, "ERROR", "", "", "", 
                        "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"
                    ])
        
        print(f"\n✅ Analysis complete!")
        print(f"   Processed: {processed_count} files successfully")
        print(f"   Errors: {error_count} files")
        print(f"   Output: {output_file}")
        
        # Show sample of output
        if processed_count > 0:
            print(f"\nSample from output file:")
            with open(output_file, "r") as f:
                for i, line in enumerate(f):
                    if i < 5:  # Show first 5 lines
                        print(f"  {line.strip()}")
                    else:
                        break
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Value error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()