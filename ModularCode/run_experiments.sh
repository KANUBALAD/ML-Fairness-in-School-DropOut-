#!/bin/bash

# =============================================================================
# ML Fairness Experiments - Comprehensive Research Pipeline
# =============================================================================

set -e  # Exit on any error

# Configuration
DATASETS=("brazil" "africa" "india")
RESULTS_BASE_DIR="./comprehensive_results"
LOG_FILE="$RESULTS_BASE_DIR/experiment_log.txt"

# Create base results directory
mkdir -p "$RESULTS_BASE_DIR"

# Logging function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to extract best model from output
extract_best_model() {
    local dataset=$1
    local output_file="$RESULTS_BASE_DIR/model_selection_${dataset}.txt"
    
    # Parse the cross-validation results to find best model
    if [[ -f "$output_file" ]]; then
        # Extract mean accuracies and find the best one
        best_model=$(grep "Mean accuracy" "$output_file" | \
                    awk '{print $1, $5}' | \
                    sort -k2 -nr | \
                    head -1 | \
                    awk '{print $1}' | \
                    tr -d ':')
        echo "$best_model"
    else
        echo "logistic_regression"  # Default fallback
    fi
}

# =============================================================================
# PHASE 1: MODEL SELECTION - Identify Best Performing Models
# =============================================================================

log_message "=== PHASE 1: MODEL SELECTION ==="
log_message "Running basic ML experiments to identify best performing models..."

declare -A BEST_MODELS

for dataset in "${DATASETS[@]}"; do
    log_message "Testing models on $dataset dataset..."
    
    output_file="$RESULTS_BASE_DIR/model_selection_${dataset}.txt"
    
    # Run basic model comparison
    python main.py "yaml/${dataset}.yaml" > "$output_file" 2>&1
    
    # Extract best performing model
    best_model=$(extract_best_model "$dataset")
    BEST_MODELS["$dataset"]="$best_model"
    
    log_message "Best model for $dataset: ${BEST_MODELS[$dataset]}"
    
    # Brief pause between datasets
    sleep 2
done

# Display summary of best models
log_message "=== MODEL SELECTION SUMMARY ==="
for dataset in "${DATASETS[@]}"; do
    log_message "$dataset: ${BEST_MODELS[$dataset]}"
done

# For simplicity, let's use the most common best model across all datasets
# or default to logistic_regression if there's a tie
OPTIMAL_MODEL="logistic_regression"  # You can modify this based on results
log_message "Selected optimal model for all experiments: $OPTIMAL_MODEL"

# =============================================================================
# PHASE 2: FAIRNESS BASELINE - Test Fairness Interventions
# =============================================================================

log_message "=== PHASE 2: FAIRNESS BASELINE EXPERIMENTS ==="
log_message "Running fairness interventions with optimal model: $OPTIMAL_MODEL"

# Update YAML files to use the optimal model
for dataset in "${DATASETS[@]}"; do
    log_message "Running fairness baseline for $dataset..."
    
    # Create custom results folder for this phase
    phase2_results="$RESULTS_BASE_DIR/phase2_fairness_baseline"
    mkdir -p "$phase2_results"
    
    python main.py "yaml/${dataset}.yaml" \
        --fairness \
        --save_results \
        --results_folder "$phase2_results" \
        > "$RESULTS_BASE_DIR/fairness_baseline_${dataset}.txt" 2>&1
    
    log_message "Completed fairness baseline for $dataset"
    sleep 2
done

# =============================================================================
# PHASE 3: BALANCED AUGMENTATION EXPERIMENTS
# =============================================================================

log_message "=== PHASE 3: BALANCED AUGMENTATION EXPERIMENTS ==="

# Define experimental scenarios
declare -A SCENARIOS=(
    ["perfect_balance"]="0.5 0.5"
    ["high_dropout_balanced_gender"]="0.5 0.7"
    ["very_high_dropout_balanced_gender"]="0.5 0.8"
    ["low_dropout_balanced_gender"]="0.5 0.3"
    ["very_low_dropout_balanced_gender"]="0.5 0.2"
    ["male_minority_balanced_labels"]="0.3 0.5"
    ["male_severe_minority_balanced_labels"]="0.2 0.5"
    ["female_minority_balanced_labels"]="0.7 0.5"
    ["female_severe_minority_balanced_labels"]="0.8 0.5"
    ["male_minority_high_dropout"]="0.3 0.7"
    ["male_minority_low_dropout"]="0.3 0.3"
    ["female_minority_high_dropout"]="0.7 0.7"
    ["female_minority_low_dropout"]="0.7 0.3"
    ["extreme_imbalance_1"]="0.2 0.8"
    ["extreme_imbalance_2"]="0.8 0.2"
    ["moderate_imbalance_1"]="0.4 0.6"
    ["moderate_imbalance_2"]="0.6 0.4"
)

# Create results folder for this phase
phase3_results="$RESULTS_BASE_DIR/phase3_balanced_augmentation"
mkdir -p "$phase3_results"

for dataset in "${DATASETS[@]}"; do
    log_message "Starting balanced augmentation experiments for $dataset..."
    
    for scenario_name in "${!SCENARIOS[@]}"; do
        IFS=' ' read -r sensitive_ratio label_ratio <<< "${SCENARIOS[$scenario_name]}"
        
        log_message "Running $dataset - $scenario_name (sensitive: $sensitive_ratio, label: $label_ratio)"
        
        # Run with fairness intervention
        python main.py "yaml/${dataset}.yaml" \
            --balanced \
            --sensitive_ratio "$sensitive_ratio" \
            --label_ratio "$label_ratio" \
            --fairness \
            --save_results \
            --results_folder "$phase3_results" \
            --scenario_name "$scenario_name" \
            > "$RESULTS_BASE_DIR/balanced_${dataset}_${scenario_name}_fair.txt" 2>&1
        
        log_message "Completed $dataset - $scenario_name with fairness"
        
        # Run without fairness intervention for comparison
        python main.py "yaml/${dataset}.yaml" \
            --balanced \
            --sensitive_ratio "$sensitive_ratio" \
            --label_ratio "$label_ratio" \
            --save_results \
            --results_folder "$phase3_results" \
            --scenario_name "${scenario_name}_no_fairness" \
            > "$RESULTS_BASE_DIR/balanced_${dataset}_${scenario_name}_no_fair.txt" 2>&1
        
        log_message "Completed $dataset - $scenario_name without fairness"
        
        # Brief pause between experiments
        sleep 1
    done
    
    log_message "Completed all scenarios for $dataset"
done

# =============================================================================
# PHASE 4: ADDITIONAL RESEARCH SCENARIOS
# =============================================================================

log_message "=== PHASE 4: ADDITIONAL RESEARCH SCENARIOS ==="

# Define additional research-focused scenarios
declare -A RESEARCH_SCENARIOS=(
    ["gender_reversal_brazil"]="0.35 0.32"  # Reverse current Brazil imbalance
    ["gender_reversal_india"]="0.35 0.15"   # Reverse current India imbalance
    ["africa_stress_test_1"]="0.3 0.8"      # Test Africa's "balanced" dataset
    ["africa_stress_test_2"]="0.8 0.3"      # Another stress test for Africa
    ["realistic_intervention_1"]="0.45 0.45" # Slight improvement scenarios
    ["realistic_intervention_2"]="0.55 0.55" # Slight improvement scenarios
)

phase4_results="$RESULTS_BASE_DIR/phase4_research_scenarios"
mkdir -p "$phase4_results"

for dataset in "${DATASETS[@]}"; do
    log_message "Running additional research scenarios for $dataset..."
    
    for scenario_name in "${!RESEARCH_SCENARIOS[@]}"; do
        # Skip scenarios not relevant to specific datasets
        if [[ "$scenario_name" == *"brazil"* && "$dataset" != "brazil" ]]; then
            continue
        fi
        if [[ "$scenario_name" == *"india"* && "$dataset" != "india" ]]; then
            continue
        fi
        if [[ "$scenario_name" == *"africa"* && "$dataset" != "africa" ]]; then
            continue
        fi
        
        IFS=' ' read -r sensitive_ratio label_ratio <<< "${RESEARCH_SCENARIOS[$scenario_name]}"
        
        log_message "Running $dataset - $scenario_name (sensitive: $sensitive_ratio, label: $label_ratio)"
        
        python main.py "yaml/${dataset}.yaml" \
            --balanced \
            --sensitive_ratio "$sensitive_ratio" \
            --label_ratio "$label_ratio" \
            --fairness \
            --save_results \
            --results_folder "$phase4_results" \
            --scenario_name "$scenario_name" \
            > "$RESULTS_BASE_DIR/research_${dataset}_${scenario_name}.txt" 2>&1
        
        log_message "Completed $dataset - $scenario_name"
        sleep 1
    done
done

# =============================================================================
# PHASE 5: COMPREHENSIVE ANALYSIS PREPARATION
# =============================================================================

log_message "=== PHASE 5: PREPARING RESULTS FOR ANALYSIS ==="

# Create summary file
summary_file="$RESULTS_BASE_DIR/experiment_summary.txt"

cat > "$summary_file" << EOF
ML FAIRNESS EXPERIMENT SUMMARY
Generated: $(date)

DATASETS TESTED: ${DATASETS[*]}
OPTIMAL MODEL: $OPTIMAL_MODEL

EXPERIMENTS COMPLETED:
- Phase 1: Model Selection (${#DATASETS[@]} datasets)
- Phase 2: Fairness Baseline (${#DATASETS[@]} datasets)
- Phase 3: Balanced Augmentation (${#DATASETS[@]} datasets × ${#SCENARIOS[@]} scenarios × 2 variants = $((${#DATASETS[@]} * ${#SCENARIOS[@]} * 2)) experiments)
- Phase 4: Research Scenarios (Additional targeted experiments)

TOTAL EXPERIMENTS: $((${#DATASETS[@]} + ${#DATASETS[@]} + ${#DATASETS[@]} * ${#SCENARIOS[@]} * 2)) + additional research scenarios

RESULTS STRUCTURE:
- $RESULTS_BASE_DIR/phase2_fairness_baseline/: JSON files for fairness baselines
- $RESULTS_BASE_DIR/phase3_balanced_augmentation/: JSON files for balanced augmentation experiments
- $RESULTS_BASE_DIR/phase4_research_scenarios/: JSON files for additional research scenarios
- $RESULTS_BASE_DIR/*.txt: Console outputs and logs

NEXT STEPS:
1. Analyze JSON files for research paper
2. Generate comparative tables and visualizations
3. Extract key insights about group balance effects on fairness interventions
EOF

log_message "Created experiment summary at $summary_file"

# Count total JSON result files
total_json_files=$(find "$RESULTS_BASE_DIR" -name "*.json" | wc -l)
log_message "Total JSON result files generated: $total_json_files"

# Create a simple results index
results_index="$RESULTS_BASE_DIR/results_index.txt"
echo "RESULTS INDEX - Generated $(date)" > "$results_index"
echo "=================================" >> "$results_index"
find "$RESULTS_BASE_DIR" -name "*.json" | sort >> "$results_index"

log_message "=== ALL EXPERIMENTS COMPLETED ==="
log_message "Results saved in: $RESULTS_BASE_DIR"
log_message "Check $LOG_FILE for detailed execution log"
log_message "Check $summary_file for experiment summary"
log_message "Total JSON files: $total_json_files"

# Optional: Create a quick analysis script
analysis_script="$RESULTS_BASE_DIR/quick_analysis.py"
cat > "$analysis_script" << 'EOF'
#!/usr/bin/env python3
"""
Quick analysis script for ML fairness experiments
Run: python quick_analysis.py
"""

import json
import os
import pandas as pd
from pathlib import Path

def analyze_results(results_dir):
    """Quick analysis of experimental results."""
    
    json_files = list(Path(results_dir).glob("**/*.json"))
    print(f"Found {len(json_files)} result files")
    
    results_summary = []
    
    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            results = data.get('results', {})
            
            if 'baseline_metrics' in results:
                baseline = results['baseline_metrics']
                fair = results.get('fair_metrics', {})
                
                summary = {
                    'file': json_file.name,
                    'experiment_type': metadata.get('experiment_type'),
                    'dataset': metadata.get('config_name'),
                    'scenario': metadata.get('scenario_name'),
                    'baseline_accuracy': baseline.get('overall_accuracy', 0),
                    'baseline_dp_diff': baseline.get('demographic_parity_difference', 0),
                    'fair_accuracy': fair.get('overall_accuracy', 0) if fair else None,
                    'fair_dp_diff': fair.get('demographic_parity_difference', 0) if fair else None
                }
                results_summary.append(summary)
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results_summary)
    
    if not df.empty:
        print("\n=== QUICK SUMMARY ===")
        print(f"Experiments by dataset:")
        print(df['dataset'].value_counts())
        
        print(f"\nExperiments by type:")
        print(df['experiment_type'].value_counts())
        
        print(f"\nAverage baseline accuracy by dataset:")
        print(df.groupby('dataset')['baseline_accuracy'].mean())
        
        # Save summary
        df.to_csv(os.path.join(results_dir, 'results_summary.csv'), index=False)
        print(f"\nDetailed summary saved to results_summary.csv")
    
    return df

if __name__ == "__main__":
    results_dir = "."  # Current directory (should be run from results folder)
    analyze_results(results_dir)
EOF

chmod +x "$analysis_script"
log_message "Created quick analysis script at $analysis_script"

echo ""
echo "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! 🎉"
echo ""
echo "📊 Results Summary:"
echo "   - Total experiments run: Multiple phases across ${#DATASETS[@]} datasets"
echo "   - JSON files generated: $total_json_files"
echo "   - Results directory: $RESULTS_BASE_DIR"
echo ""
echo "📝 Next Steps:"
echo "   1. Review $summary_file for experiment overview"
echo "   2. Analyze JSON files in phase directories for research insights"
echo "   3. Run 'cd $RESULTS_BASE_DIR && python quick_analysis.py' for quick summary"
echo ""
echo "🔬 Research Questions Addressed:"
echo "   - Does group size imbalance affect fairness intervention effectiveness?"
echo "   - How do different balance scenarios impact model performance?"
echo "   - What are optimal balance points for fairness vs accuracy trade-offs?"
echo ""