# experiments/analyze_results_from_csv.py
import pandas as pd
import numpy as np
import sys

def analyze_csv(csv_path):
    print(f"\n--- 🔬 Deep Dive Analysis for: {csv_path} ---\n")
    
    try:
        # Load CSV with detailed error handling
        df = pd.read_csv(csv_path, na_values=['N/A', 'NA', 'NaN', ''])
        print(f"CSV loaded successfully with {len(df)} rows")
        
        # Debug: Show column names and types
        print("\nColumns and Data Types:")
        print(df.dtypes)
        
        # Convert numeric columns
        numeric_cols = ['DP_Improvement', 'Accuracy_Cost']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Converted {col} to numeric. Missing values: {df[col].isna().sum()}")
        
        # Filter valid rows
        valid_df = df[~df['DP_Improvement'].isna()]
        print(f"\nFound {len(valid_df)} valid rows after filtering")
        
        if valid_df.empty:
            print("No valid rows to analyze. Possible reasons:")
            print("- All DP_Improvement values are missing")
            print("- Data type conversion failed")
            print("- CSV structure doesn't match expectations")
            print("\nFirst 5 rows of original data:")
            print(df.head())
            return

        # Find baselines
        baselines = {}
        for dataset in valid_df['Dataset'].unique():
            baseline = valid_df[
                (valid_df['Dataset'] == dataset) & 
                (valid_df['Scenario'] == 'original_ratios')
            ]
            if not baseline.empty:
                baselines[dataset] = baseline['DP_Improvement'].values[0]
                print(f"Baseline for {dataset}: {baselines[dataset]}")
            else:
                print(f"⚠️ Warning: No baseline found for {dataset}")
                baselines[dataset] = 0  # Default to 0 if missing

        # Calculate bias amplification
        results = []
        for _, row in valid_df.iterrows():
            dataset = row['Dataset']
            baseline_dp = baselines.get(dataset, 0)
            final_dp = baseline_dp + row['DP_Improvement']
            bias_amplified = abs(final_dp) > abs(baseline_dp)
            
            results.append({
                "Dataset": dataset,
                "Scenario": row['Scenario'],
                "Baseline DP": baseline_dp,
                "Final DP": final_dp,
                "Bias Amplified?": "🔥 YES" if bias_amplified else "No",
                "DP Improvement": row['DP_Improvement'],
                "Accuracy Cost": row['Accuracy_Cost']
            })

        # Create results dataframe
        results_df = pd.DataFrame(results)
        print("\n--- Bias Amplification Analysis Results ---")
        print(results_df)
        
        # Top amplified scenarios
        amplified_df = results_df[results_df['Bias Amplified?'] == "🔥 YES"]
        if not amplified_df.empty:
            amplified_df = amplified_df.copy()
            amplified_df['Amplification Factor'] = amplified_df.apply(
                lambda x: abs(x['Final DP']) - abs(x['Baseline DP']), 
                axis=1
            )
            amplified_df = amplified_df.sort_values('Amplification Factor', ascending=False)
            
            print("\n🔥 TOP 5 BIAS AMPLIFICATION SCENARIOS")
            print(amplified_df[['Dataset', 'Scenario', 'Amplification Factor']].head(5))
        else:
            print("\n✅ No scenarios amplified bias")
            
    except Exception as e:
        print(f"🚨 Critical error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_results_from_csv.py <path_to_csv>")
        sys.exit(1)
    
    analyze_csv(sys.argv[1])