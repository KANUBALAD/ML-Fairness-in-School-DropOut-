# experiments/analyze_results_from_csv.py
import pandas as pd
import numpy as np

def analyze_csv(csv_path):
    print(f"\n--- 🔬 Deep Dive Analysis for: {csv_path} ---\n")
    
    # Load CSV with explicit handling of 'N/A'
    df = pd.read_csv(csv_path, na_values=['N/A'])
    
    # Filter valid rows (DP_Improvement must exist)
    valid_df = df[~df['DP_Improvement'].isna()].copy()
    
    if valid_df.empty:
        print("Could not find any valid, complete rows in the CSV to analyze.")
        return

    # Calculate baseline DP for each dataset
    baselines = {}
    for dataset in valid_df['Dataset'].unique():
        baseline_row = valid_df[
            (valid_df['Dataset'] == dataset) & 
            (valid_df['Scenario'] == 'original_ratios')
        ]
        if not baseline_row.empty:
            baselines[dataset] = baseline_row['DP_Improvement'].values[0]

    # Calculate bias amplification
    results = []
    for _, row in valid_df.iterrows():
        dataset = row['Dataset']
        baseline_dp = baselines.get(dataset, np.nan)
        
        if not np.isnan(baseline_dp):
            final_dp = row['DP_Improvement'] + baseline_dp
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

    # Create result dataframe
    results_df = pd.DataFrame(results)
    print("--- Bias Amplification Analysis Results ---\n")
    print(results_df)
    
    # Top 5 amplified scenarios
    print("\n\n--- 🔥 TOP 5 BIAS AMPLIFICATION SCENARIOS ---")
    amplified_df = results_df[results_df['Bias Amplified?'] == "🔥 YES"]
    amplified_df = amplified_df.sort_values(
        by=lambda x: abs(x['Final DP']) - abs(x['Baseline DP']), 
        ascending=False
    )
    print(amplified_df.head(5))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_results_from_csv.py <path_to_csv>")
        sys.exit(1)
    
    analyze_csv(sys.argv[1])