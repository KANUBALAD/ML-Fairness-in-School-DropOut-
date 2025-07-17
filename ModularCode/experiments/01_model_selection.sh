#!/bin/bash

# =============================================================================
# EXPERIMENT 1: MODEL SELECTION
# Find the best performing model for each dataset
# =============================================================================

set -e

# Configuration
DATASETS=("brazil" "africa" "india")
EXPERIMENT_NAME="01_model_selection"
RESULTS_DIR="./experiments/results/$EXPERIMENT_NAME"
LOG_FILE="$RESULTS_DIR/experiment.log"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "./experiments/utils"

# Logging function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_message "=== EXPERIMENT 1: MODEL SELECTION ==="
log_message "Objective: Identify best performing model for each dataset"
log_message "Method: 5x2 cross-validation comparison across logistic_regression, decision_tree, random_forest"

# Create results summary file
SUMMARY_FILE="$RESULTS_DIR/model_selection_summary.txt"
echo "MODEL SELECTION SUMMARY - $(date)" > "$SUMMARY_FILE"
echo "=======================================" >> "$SUMMARY_FILE"

declare -A BEST_MODELS

for dataset in "${DATASETS[@]}"; do
    log_message "Testing models on $dataset dataset..."
    
    # Run model comparison with cross-validation
    output_file="$RESULTS_DIR/${dataset}_model_comparison.txt"
    
    python main.py "yaml/${dataset}.yaml" > "$output_file" 2>&1
    
    # Extract best model (we'll create a Python helper for this)
    best_model=$(python experiments/utils/extract_best_model.py "$output_file")
    BEST_MODELS["$dataset"]="$best_model"
    
    # Log results
    log_message "Best model for $dataset: $best_model"
    echo "$dataset: $best_model" >> "$SUMMARY_FILE"
    
    # Add detailed results to summary
    echo "" >> "$SUMMARY_FILE"
    echo "=== $dataset Detailed Results ===" >> "$SUMMARY_FILE"
    grep -A 10 -B 2 "Mean accuracy" "$output_file" >> "$SUMMARY_FILE" || true
    echo "" >> "$SUMMARY_FILE"
    
    sleep 2
done

# Create machine-readable results file
RESULTS_JSON="$RESULTS_DIR/best_models.json"
cat > "$RESULTS_JSON" << EOF
{
    "experiment": "model_selection",
    "timestamp": "$(date -Iseconds)",
    "best_models": {
        "brazil": "${BEST_MODELS[brazil]}",
        "africa": "${BEST_MODELS[africa]}",
        "india": "${BEST_MODELS[india]}"
    },
    "methodology": "5x2_cross_validation",
    "models_tested": ["logistic_regression", "decision_tree", "random_forest"]
}
EOF

log_message "=== RESULTS SUMMARY ==="
for dataset in "${DATASETS[@]}"; do
    log_message "$dataset: ${BEST_MODELS[$dataset]}"
done

log_message "Results saved to: $RESULTS_DIR"
log_message "Summary: $SUMMARY_FILE"
log_message "Machine-readable: $RESULTS_JSON"

echo ""
echo "🎯 EXPERIMENT 1 COMPLETED"
echo "📊 Results:"
for dataset in "${DATASETS[@]}"; do
    echo "   $dataset: ${BEST_MODELS[$dataset]}"
done
echo ""
echo "📁 Next Steps:"
echo "   1. Review results in: $RESULTS_DIR"
echo "   2. Run: bash experiments/02_baseline_fairness.sh"
echo ""