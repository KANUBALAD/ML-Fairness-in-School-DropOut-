#!/bin/bash

# =============================================================================
# EXPERIMENT 2: BASELINE FAIRNESS ASSESSMENT
# Establish fairness baselines using best models from Experiment 1
# =============================================================================

set -e

# Configuration
DATASETS=("brazil" "africa" "india")
EXPERIMENT_NAME="02_baseline_fairness"
RESULTS_DIR="./experiments/results/$EXPERIMENT_NAME"
LOG_FILE="$RESULTS_DIR/experiment.log"
PREV_RESULTS="./experiments/results/01_model_selection/best_models.json"

# Create directories
mkdir -p "$RESULTS_DIR"

# Check if previous experiment results exist
if [[ ! -f "$PREV_RESULTS" ]]; then
    echo "❌ Error: Run experiments/01_model_selection.sh first!"
    echo "   Missing: $PREV_RESULTS"
    exit 1
fi

# Load best models from previous experiment
source <(python -c "
import json
with open('$PREV_RESULTS') as f:
    data = json.load(f)
    best_models = data['best_models']
    for dataset, model in best_models.items():
        print(f'BEST_MODEL_{dataset.upper()}={model}')
")

# Logging function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_message "=== EXPERIMENT 2: BASELINE FAIRNESS ASSESSMENT ==="
log_message "Objective: Establish fairness baselines for each dataset using optimal models"
log_message "Previous results loaded from: $PREV_RESULTS"
log_message "Best models: Brazil=${BEST_MODEL_BRAZIL}, Africa=${BEST_MODEL_AFRICA}, India=${BEST_MODEL_INDIA}"

# Create summary file
SUMMARY_FILE="$RESULTS_DIR/baseline_fairness_summary.txt"
echo "BASELINE FAIRNESS ASSESSMENT - $(date)" > "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"

# Function to update YAML with best model
update_yaml_model() {
    local dataset=$1
    local model=$2
    local yaml_file="yaml/${dataset}.yaml"
    
    # Create a temporary yaml with the best model
    local temp_yaml="$RESULTS_DIR/temp_${dataset}.yaml"
    
    # Copy original and update model field
    cp "$yaml_file" "$temp_yaml"
    
    # Update the model field (assumes YAML has 'model: compare' or similar)
    if command -v yq >/dev/null 2>&1; then
        yq eval ".model = \"$model\"" -i "$temp_yaml"
    else
        # Fallback: simple sed replacement
        sed -i.bak "s/model: .*/model: $model/" "$temp_yaml"
    fi
    
    echo "$temp_yaml"
}

for dataset in "${DATASETS[@]}"; do
    # Get best model for this dataset
    best_model_var="BEST_MODEL_${dataset^^}"
    best_model="${!best_model_var}"
    
    log_message "Running baseline fairness assessment for $dataset using $best_model..."
    
    # Update YAML to use best model
    temp_yaml=$(update_yaml_model "$dataset" "$best_model")
    
    # Run baseline experiment (no augmentation, with fairness)
    python main.py "$temp_yaml" \
        --fairness \
        --save_results \
        --results_folder "$RESULTS_DIR" \
        > "$RESULTS_DIR/${dataset}_baseline_output.txt" 2>&1
    
    log_message "Completed baseline assessment for $dataset"
    
    # Extract key metrics for summary
    echo "=== $dataset Baseline (Model: $best_model) ===" >> "$SUMMARY_FILE"
    
    # Look for fairness report in output
    if grep -q "Fairness Report" "$RESULTS_DIR/${dataset}_baseline_output.txt"; then
        echo "Fairness metrics found - extracting summary..." >> "$SUMMARY_FILE"
        grep -A 20 "Fairness Report" "$RESULTS_DIR/${dataset}_baseline_output.txt" | head -25 >> "$SUMMARY_FILE"
    else
        echo "No fairness metrics found in output." >> "$SUMMARY_FILE"
    fi
    
    echo "" >> "$SUMMARY_FILE"
    
    # Clean up temp file
    rm -f "$temp_yaml" "${temp_yaml}.bak"
    
    sleep 2
done

# Create machine-readable baseline results
BASELINE_JSON="$RESULTS_DIR/baseline_results.json"
python experiments/utils/analyze_results.py "$RESULTS_DIR" --output "$BASELINE_JSON" --experiment "baseline_fairness"

log_message "=== BASELINE FAIRNESS ASSESSMENT COMPLETED ==="
log_message "Results saved to: $RESULTS_DIR"
log_message "Summary: $SUMMARY_FILE"
log_message "Analysis: $BASELINE_JSON"

echo ""
echo "📊 EXPERIMENT 2 COMPLETED"
echo "🎯 Objective: Baseline fairness established for all datasets"
echo "📁 Results: $RESULTS_DIR"
echo ""
echo "📋 Key Findings:"
echo "   - Review fairness metrics in $SUMMARY_FILE"
echo "   - Check detailed JSON results in $RESULTS_DIR"
echo ""
echo "📁 Next Steps:"
echo "   1. Analyze baseline fairness patterns"
echo "   2. Run: bash experiments/03_imbalance_impact.sh"
echo ""