#!/usr/bin/env python3
"""
Comprehensive analysis utility for ML fairness experiments.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from collections import defaultdict

def convert_pandas_types(obj):
    """Convert pandas/numpy types to JSON serializable types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_pandas_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_pandas_types(item) for item in obj]
    else:
        return obj

def analyze_results(results_dir, output_file=None, experiment_type=None, compare_scenarios=False, compare_methods=False):
    """Comprehensive analysis of experimental results."""
    
    json_files = list(Path(results_dir).glob("**/*.json"))
    print(f"Found {len(json_files)} result files in {results_dir}")
    
    if len(json_files) == 0:
        print("No JSON files found. Make sure experiments have completed successfully.")
        return
    
    results_data = []
    
    # Load all results
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            results = data.get('results', {})
            
            # Extract key metrics
            if 'baseline_metrics' in results and 'fair_metrics' in results:
                baseline = results['baseline_metrics']
                fair = results['fair_metrics']
                improvement = results.get('improvement_metrics', {})
                
                result_entry = {
                    'file': json_file.name,
                    'experiment_type': metadata.get('experiment_type'),
                    'dataset': metadata.get('config_name'),
                    'scenario': metadata.get('scenario_name'),
                    'method': extract_method_from_scenario(metadata.get('scenario_name', '')),
                    'baseline_accuracy': baseline.get('overall_accuracy', 0),
                    'baseline_dp_diff': baseline.get('demographic_parity_difference', 0),
                    'fair_accuracy': fair.get('overall_accuracy', 0),
                    'fair_dp_diff': fair.get('demographic_parity_difference', 0),
                    'dp_improvement': improvement.get('dp_improvement', 0),
                    'accuracy_cost': improvement.get('accuracy_difference', 0),
                    'balance_achieved': results.get('balance_achieved', {}),
                    'generation_method': results.get('generation_method', 'unknown'),
                    'privileged_accuracy': baseline.get('privileged_accuracy', 0),
                    'unprivileged_accuracy': baseline.get('unprivileged_accuracy', 0),
                    'fair_privileged_accuracy': fair.get('privileged_accuracy', 0),
                    'fair_unprivileged_accuracy': fair.get('unprivileged_accuracy', 0),
                    'tpr_improvement': improvement.get('tpr_improvement', 0),
                    'fpr_improvement': improvement.get('fpr_improvement', 0)
                }
                results_data.append(result_entry)
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if not results_data:
        print("No valid result data found.")
        return
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results_data)
    
    # Generate analysis based on experiment type
    analysis = {
        'summary': generate_summary_analysis(df),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    if experiment_type == "imbalance_impact" or compare_scenarios:
        analysis['imbalance_analysis'] = analyze_imbalance_impact(df)
    
    if experiment_type == "method_comparison" or compare_methods:
        analysis['method_comparison'] = analyze_method_effectiveness(df)
    
    # Convert all pandas types before saving
    analysis = convert_pandas_types(analysis)
    
    # Save detailed CSV
    csv_file = Path(results_dir) / "detailed_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to: {csv_file}")
    
    # Save analysis JSON
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"Analysis saved to: {output_file}")
    
    # Print summary
    print_analysis_summary(analysis)
    
    return analysis

def extract_method_from_scenario(scenario_name):
    """Extract method from scenario name."""
    methods = ['faker', 'ctgan', 'tvae', 'llm_async']
    for method in methods:
        if method in scenario_name.lower():
            return method
    return 'faker'  # Default assumption

def generate_summary_analysis(df):
    """Generate summary statistics."""
    return {
        'total_experiments': int(len(df)),
        'datasets': int(df['dataset'].nunique()),
        'scenarios': int(df['scenario'].nunique()),
        'avg_baseline_accuracy': float(df['baseline_accuracy'].mean()),
        'avg_fair_accuracy': float(df['fair_accuracy'].mean()),
        'avg_dp_improvement': float(df['dp_improvement'].mean()),
        'avg_accuracy_cost': float(df['accuracy_cost'].mean()),
        'successful_interventions': int((df['dp_improvement'] > 0.03).sum()),
        'failed_interventions': int((df['dp_improvement'] <= 0).sum())
    }

def analyze_imbalance_impact(df):
    """Analyze how imbalance affects fairness interventions."""
    imbalance_analysis = {}
    
    # Group by dataset and analyze patterns
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        # Find best and worst scenarios
        best_idx = dataset_df['dp_improvement'].idxmax()
        worst_idx = dataset_df['dp_improvement'].idxmin()
        
        # Count successful scenarios (>3% improvement)
        successful_scenarios = dataset_df[dataset_df['dp_improvement'] > 0.03]
        failed_scenarios = dataset_df[dataset_df['dp_improvement'] <= 0]
        
        imbalance_analysis[dataset] = {
            'best_scenario': {
                'name': str(dataset_df.loc[best_idx, 'scenario']),
                'dp_improvement': float(dataset_df.loc[best_idx, 'dp_improvement']),
                'accuracy_cost': float(dataset_df.loc[best_idx, 'accuracy_cost']),
                'method': str(dataset_df.loc[best_idx, 'method'])
            },
            'worst_scenario': {
                'name': str(dataset_df.loc[worst_idx, 'scenario']),
                'dp_improvement': float(dataset_df.loc[worst_idx, 'dp_improvement']),
                'accuracy_cost': float(dataset_df.loc[worst_idx, 'accuracy_cost']),
                'method': str(dataset_df.loc[worst_idx, 'method'])
            },
            'avg_dp_improvement': float(dataset_df['dp_improvement'].mean()),
            'successful_scenarios_count': int(len(successful_scenarios)),
            'failed_scenarios_count': int(len(failed_scenarios)),
            'total_scenarios': int(len(dataset_df)),
            'success_rate': float(len(successful_scenarios) / len(dataset_df)),
            'breakthrough_scenarios': [
                {
                    'name': str(row['scenario']),
                    'dp_improvement': float(row['dp_improvement']),
                    'method': str(row['method'])
                }
                for _, row in successful_scenarios.iterrows()
            ]
        }
    
    return imbalance_analysis

def analyze_method_effectiveness(df):
    """Analyze effectiveness of different synthetic data methods."""
    method_analysis = {}
    
    methods = df['generation_method'].unique()
    
    for method in methods:
        method_df = df[df['generation_method'] == method]
        
        if len(method_df) > 0:
            best_idx = method_df['dp_improvement'].idxmax()
            
            method_analysis[method] = {
                'avg_dp_improvement': float(method_df['dp_improvement'].mean()),
                'avg_accuracy_cost': float(method_df['accuracy_cost'].mean()),
                'success_rate': float((method_df['dp_improvement'] > 0.03).mean()),
                'total_experiments': int(len(method_df)),
                'best_case': {
                    'dataset': str(method_df.loc[best_idx, 'dataset']),
                    'scenario': str(method_df.loc[best_idx, 'scenario']),
                    'dp_improvement': float(method_df.loc[best_idx, 'dp_improvement'])
                },
                'reliability_score': float(calculate_reliability_score(method_df))
            }
    
    # Rank methods
    method_rankings = rank_methods(method_analysis)
    method_analysis['rankings'] = method_rankings
    
    return method_analysis

def calculate_reliability_score(method_df):
    """Calculate a reliability score for a method."""
    if len(method_df) == 0:
        return 0.0
    
    positive_impact_rate = (method_df['dp_improvement'] > 0).mean()
    consistency = 1 - (method_df['dp_improvement'].std() / (abs(method_df['dp_improvement'].mean()) + 0.001))
    
    return (positive_impact_rate * 0.7 + consistency * 0.3)

def rank_methods(method_analysis):
    """Rank methods by effectiveness."""
    methods = [m for m in method_analysis.keys() if m != 'rankings']
    
    rankings = {
        'by_dp_improvement': sorted(methods, 
                                  key=lambda m: method_analysis[m]['avg_dp_improvement'], 
                                  reverse=True),
        'by_success_rate': sorted(methods, 
                                key=lambda m: method_analysis[m]['success_rate'], 
                                reverse=True),
        'by_reliability': sorted(methods, 
                               key=lambda m: method_analysis[m]['reliability_score'], 
                               reverse=True)
    }
    
    return rankings

def print_analysis_summary(analysis):
    """Print a formatted summary of the analysis."""
    print("\n" + "="*60)
    print("EXPERIMENT ANALYSIS SUMMARY")
    print("="*60)
    
    summary = analysis['summary']
    print(f"Total Experiments: {summary['total_experiments']}")
    print(f"Datasets: {summary['datasets']}")
    print(f"Successful Interventions (>3% DP improvement): {summary['successful_interventions']}")
    print(f"Failed Interventions (≤0% DP improvement): {summary['failed_interventions']}")
    print(f"Average DP Improvement: {summary['avg_dp_improvement']:.4f}")
    print(f"Average Accuracy Cost: {summary['avg_accuracy_cost']:.4f}")
    
    if 'imbalance_analysis' in analysis:
        print("\nIMBALANCE IMPACT BY DATASET:")
        for dataset, data in analysis['imbalance_analysis'].items():
            print(f"\n{dataset.upper()}:")
            print(f"  Success rate: {data['success_rate']:.1%}")
            print(f"  Best scenario: {data['best_scenario']['name']} (+{data['best_scenario']['dp_improvement']:.3f})")
            print(f"  Avg DP improvement: {data['avg_dp_improvement']:.4f}")
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ML fairness experiment results")
    parser.add_argument("results_dir", help="Directory containing result JSON files")
    parser.add_argument("--output", help="Output file for analysis JSON")
    parser.add_argument("--experiment", help="Experiment type for specialized analysis")
    parser.add_argument("--compare-scenarios", action="store_true", help="Include scenario comparison")
    parser.add_argument("--compare-methods", action="store_true", help="Include method comparison")
    
    args = parser.parse_args()
    
    analyze_results(
        args.results_dir,
        args.output,
        args.experiment,
        args.compare_scenarios,
        args.compare_methods
    )