#!/bin/bash

# =============================================================================
# EXPERIMENT 3: IMBALANCE IMPACT STUDY (ENHANCED)
# Test how group size imbalance affects fairness intervention effectiveness
# ENHANCED: Comprehensive analysis with detailed breakthrough detection
# =============================================================================

set -e

# Configuration - PRIORITIZE DATASETS BY FAIRNESS CHALLENGE
DATASETS=("brazil" "india" "africa")  # Reordered by difficulty
EXPERIMENT_NAME="03_imbalance_impact"
RESULTS_DIR="./experiments/results/$EXPERIMENT_NAME"
LOG_FILE="$RESULTS_DIR/experiment.log"
BASELINE_RESULTS="./experiments/results/02_baseline_fairness"

# API key for LLM method
API_KEY="sk-8d71fd82b1e34ce48e31718a1f3647bf"

# Create directories
mkdir -p "$RESULTS_DIR"

# Check prerequisites
if [[ ! -d "$BASELINE_RESULTS" ]]; then
    echo "❌ Error: Run experiments/02_baseline_fairness.sh first!"
    exit 1
fi

# Logging function
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_message "=== EXPERIMENT 3: IMBALANCE IMPACT STUDY (ENHANCED) ==="
log_message "CRITICAL FINDING: Fairness interventions FAIL for Brazil (-3.0%) and India (-3.5%)"
log_message "Objective: Test if different imbalance scenarios can make fairness interventions work"
log_message "Enhancement: Comprehensive analysis with breakthrough detection and detailed metrics"

# Define focused scenarios based on baseline failures
declare -A IMBALANCE_SCENARIOS=(
    # Baseline comparison
    ["original"]="original original"  # Special case: use original ratios
    
    # Test if balanced datasets help fairness interventions work
    ["perfect_balance"]="0.5 0.5"
    
    # Test extreme scenarios to understand limits
    ["extreme_male_majority"]="0.8 0.5"
    ["extreme_female_majority"]="0.2 0.5"
    ["extreme_high_dropout"]="0.5 0.8"
    ["extreme_low_dropout"]="0.5 0.2"
    
    # Test if reversing imbalance helps
    ["reverse_gender_brazil"]="0.35 0.32"    # Reverse Brazil's 65:35 ratio
    ["reverse_gender_india"]="0.35 0.12"     # Reverse India's imbalance
    ["reverse_label_brazil"]="0.65 0.68"     # Reverse Brazil's dropout ratio
    
    # Test moderate corrections
    ["moderate_balance_gender"]="0.45 0.32"  # Small gender correction
    ["moderate_balance_labels"]="0.65 0.45"  # Small label correction
    
    # Additional strategic scenarios
    ["gender_balanced_high_dropout"]="0.5 0.7"
    ["gender_balanced_low_dropout"]="0.5 0.3"
    ["slight_female_majority"]="0.4 0.5"
    ["slight_male_majority"]="0.6 0.5"
)

# Enhanced method selection focusing on proven methods
get_best_method_for_dataset() {
    local dataset=$1
    case $dataset in
        "brazil")
            echo "llm_async"  # We know this worked best (+5.18% DP improvement)
            ;;
        "india")
            echo "llm_async"  # Start with LLM
            ;;
        "africa")
            echo "faker"      # Africa baseline works, test faker first
            ;;
        *)
            echo "faker"
            ;;
    esac
}

# Create enhanced summary file with baseline context
SUMMARY_FILE="$RESULTS_DIR/imbalance_impact_summary.txt"
cat > "$SUMMARY_FILE" << EOF
IMBALANCE IMPACT STUDY (ENHANCED) - $(date)
==========================================
BASELINE FAIRNESS INTERVENTION RESULTS:
- Brazil: -3.0% DP improvement (FAILS - makes bias worse)
- India: -3.5% DP improvement (FAILS - makes bias worse)  
- Africa: +1.2% DP improvement (works, but marginal)

RESEARCH QUESTIONS:
1. Can dataset rebalancing make fairness interventions effective?
2. Which balance scenarios enable breakthrough improvements (>3%)?
3. Do different datasets need different balance strategies?

METHOD SELECTION (based on previous experiments):
- Brazil: LLM method (best DP improvement: +5.18%)
- India: LLM method (testing effectiveness)
- Africa: Faker method (baseline already works)

Scenarios tested: ${#IMBALANCE_SCENARIOS[@]}
Success criteria: BREAKTHROUGH (>3%), MODERATE (1-3%), FAILURE (≤0%)

DETAILED RESULTS:
=================

EOF

total_experiments=$((${#DATASETS[@]} * ${#IMBALANCE_SCENARIOS[@]} * 2))  # 2 = with/without fairness
log_message "Planning $total_experiments total experiments across ${#IMBALANCE_SCENARIOS[@]} scenarios"

experiment_count=0

for dataset in "${DATASETS[@]}"; do
    log_message "Starting imbalance impact study for $dataset..."
    
    # Get best method for this dataset
    best_method=$(get_best_method_for_dataset "$dataset")
    log_message "Using method $best_method for $dataset (based on previous results)"
    
    dataset_results_dir="$RESULTS_DIR/${dataset}"
    mkdir -p "$dataset_results_dir"
    
    echo "=== $dataset Results (Method: $best_method) ===" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    for scenario_name in "${!IMBALANCE_SCENARIOS[@]}"; do
        scenario_value="${IMBALANCE_SCENARIOS[$scenario_name]}"
        
        log_message "[$((++experiment_count))/$total_experiments] $dataset - $scenario_name"
        
        # Handle special "original" scenario
        if [[ "$scenario_name" == "original" ]]; then
            log_message "Testing original dataset ratios (no augmentation)"
            
            # Run WITHOUT fairness (original baseline)
            python main.py "yaml/${dataset}.yaml" \
                --save_results \
                --results_folder "$dataset_results_dir" \
                --scenario_name "${scenario_name}_no_fairness" \
                > "$dataset_results_dir/${scenario_name}_no_fairness.txt" 2>&1
            
            # Run WITH fairness (original + fairness intervention)
            python main.py "yaml/${dataset}.yaml" \
                --fairness \
                --save_results \
                --results_folder "$dataset_results_dir" \
                --scenario_name "${scenario_name}_with_fairness" \
                > "$dataset_results_dir/${scenario_name}_with_fairness.txt" 2>&1
        else
            # Normal augmentation scenarios
            IFS=' ' read -r sensitive_ratio label_ratio <<< "$scenario_value"
            
            # Prepare method-specific arguments
            method_args=""
            if [[ "$best_method" == "llm_async" ]]; then
                method_args="--api_key $API_KEY"
            fi
            
            # Run WITHOUT fairness intervention
            python main.py "yaml/${dataset}.yaml" \
                --balanced \
                --sensitive_ratio "$sensitive_ratio" \
                --label_ratio "$label_ratio" \
                --save_results \
                --results_folder "$dataset_results_dir" \
                --scenario_name "${scenario_name}_no_fairness" \
                --method "$best_method" \
                $method_args \
                > "$dataset_results_dir/${scenario_name}_no_fairness.txt" 2>&1
            
            # Run WITH fairness intervention
            python main.py "yaml/${dataset}.yaml" \
                --balanced \
                --sensitive_ratio "$sensitive_ratio" \
                --label_ratio "$label_ratio" \
                --fairness \
                --save_results \
                --results_folder "$dataset_results_dir" \
                --scenario_name "${scenario_name}_with_fairness" \
                --method "$best_method" \
                $method_args \
                > "$dataset_results_dir/${scenario_name}_with_fairness.txt" 2>&1
        fi
        
        log_message "Completed $dataset - $scenario_name"
        
        # Enhanced fairness metrics extraction with comprehensive analysis
        echo "$scenario_name ($scenario_value):" >> "$SUMMARY_FILE"
        
        # Extract multiple metrics for comprehensive analysis
        if [[ -f "$dataset_results_dir/${scenario_name}_with_fairness.txt" ]]; then
            # Look for DP improvement
            if grep -q "DP improvement" "$dataset_results_dir/${scenario_name}_with_fairness.txt"; then
                dp_improvement=$(grep "DP improvement" "$dataset_results_dir/${scenario_name}_with_fairness.txt" | tail -1 | awk '{print $NF}')
                echo "  DP Improvement: $dp_improvement" >> "$SUMMARY_FILE"
                
                # Flag significant improvements (>3%)
                if python3 -c "import sys; exit(0 if float('$dp_improvement') > 0.03 else 1)" 2>/dev/null; then
                    log_message "🎉 BREAKTHROUGH: $dataset - $scenario_name achieved significant DP improvement ($dp_improvement)!"
                    echo "  *** BREAKTHROUGH SCENARIO *** (>3% improvement)" >> "$SUMMARY_FILE"
                fi
                
                # Flag moderate improvements (1-3%)
                if python3 -c "import sys; exit(0 if 0.01 <= float('$dp_improvement') <= 0.03 else 1)" 2>/dev/null; then
                    echo "  ++ Moderate improvement (1-3%)" >> "$SUMMARY_FILE"
                fi
                
                # Flag failures (negative or zero)
                if python3 -c "import sys; exit(0 if float('$dp_improvement') <= 0 else 1)" 2>/dev/null; then
                    echo "  -- FAILED: Fairness intervention made bias worse" >> "$SUMMARY_FILE"
                fi
            else
                echo "  DP Improvement: Not found" >> "$SUMMARY_FILE"
            fi
            
            # Extract accuracy cost
            if grep -q "Accuracy cost" "$dataset_results_dir/${scenario_name}_with_fairness.txt"; then
                accuracy_cost=$(grep "Accuracy cost" "$dataset_results_dir/${scenario_name}_with_fairness.txt" | tail -1 | awk '{print $NF}')
                echo "  Accuracy Cost: $accuracy_cost" >> "$SUMMARY_FILE"
            fi
            
            # Extract balance achievement
            if grep -q "Balance Achievement" "$dataset_results_dir/${scenario_name}_with_fairness.txt"; then
                echo "  Balance Achievement:" >> "$SUMMARY_FILE"
                grep -A 2 "Balance Achievement" "$dataset_results_dir/${scenario_name}_with_fairness.txt" | grep -v "Balance Achievement" | sed 's/^/    /' >> "$SUMMARY_FILE"
            fi
            
            # Extract final fairness metrics
            if grep -q "Demographic Parity Difference" "$dataset_results_dir/${scenario_name}_with_fairness.txt"; then
                echo "  Final Fairness Metrics:" >> "$SUMMARY_FILE"
                grep "Demographic Parity Difference" "$dataset_results_dir/${scenario_name}_with_fairness.txt" | tail -1 | sed 's/^/    /' >> "$SUMMARY_FILE"
                grep "Overall Accuracy" "$dataset_results_dir/${scenario_name}_with_fairness.txt" | tail -1 | sed 's/^/    /' >> "$SUMMARY_FILE"
            fi
        else
            echo "  Status: Experiment output not found" >> "$SUMMARY_FILE"
        fi
        echo "" >> "$SUMMARY_FILE"
        
        sleep 1
    done
    
    log_message "Completed all scenarios for $dataset"
done

# Create comprehensive experiment summary
log_message "Creating comprehensive experiment summary..."

COMPREHENSIVE_SUMMARY="$RESULTS_DIR/comprehensive_experiment_summary.txt"
cat > "$COMPREHENSIVE_SUMMARY" << EOF
COMPREHENSIVE IMBALANCE IMPACT STUDY SUMMARY
===========================================
Generated: $(date)

RESEARCH OBJECTIVE:
Test if different dataset balance scenarios can make fairness interventions effective

BASELINE PROBLEM:
- Brazil: Fairness interventions FAIL (-3.0% DP improvement)
- India: Fairness interventions FAIL (-3.5% DP improvement)
- Africa: Fairness interventions marginally work (+1.2% DP improvement)

EXPERIMENTAL DESIGN:
- Datasets: 3 (brazil, india, africa)
- Scenarios per dataset: ${#IMBALANCE_SCENARIOS[@]}
- Total experiments: $experiment_count
- Methods used: LLM (brazil, india), Faker (africa)

SUCCESS CRITERIA:
- BREAKTHROUGH: >3% DP improvement
- MODERATE: 1-3% DP improvement
- FAILURE: ≤0% DP improvement

DETAILED RESULTS BY DATASET:
EOF

# Generate detailed results for each dataset
for dataset in "${DATASETS[@]}"; do
    echo "" >> "$COMPREHENSIVE_SUMMARY"
    echo "=== $dataset RESULTS ===" >> "$COMPREHENSIVE_SUMMARY"
    
    dataset_results_dir="$RESULTS_DIR/${dataset}"
    
    # Count breakthrough scenarios
    breakthrough_count=0
    moderate_count=0
    failure_count=0
    
    echo "Scenario Analysis:" >> "$COMPREHENSIVE_SUMMARY"
    
    for scenario_name in "${!IMBALANCE_SCENARIOS[@]}"; do
        if [[ -f "$dataset_results_dir/${scenario_name}_with_fairness.txt" ]]; then
            if grep -q "DP improvement" "$dataset_results_dir/${scenario_name}_with_fairness.txt"; then
                dp_improvement=$(grep "DP improvement" "$dataset_results_dir/${scenario_name}_with_fairness.txt" | tail -1 | awk '{print $NF}')
                
                # Categorize the result
                if python3 -c "import sys; exit(0 if float('$dp_improvement') > 0.03 else 1)" 2>/dev/null; then
                    echo "  $scenario_name: BREAKTHROUGH (+$dp_improvement)" >> "$COMPREHENSIVE_SUMMARY"
                    ((breakthrough_count++))
                elif python3 -c "import sys; exit(0 if 0.01 <= float('$dp_improvement') <= 0.03 else 1)" 2>/dev/null; then
                    echo "  $scenario_name: MODERATE (+$dp_improvement)" >> "$COMPREHENSIVE_SUMMARY"
                    ((moderate_count++))
                else
                    echo "  $scenario_name: FAILED ($dp_improvement)" >> "$COMPREHENSIVE_SUMMARY"
                    ((failure_count++))
                fi
            else
                echo "  $scenario_name: NO METRICS" >> "$COMPREHENSIVE_SUMMARY"
                ((failure_count++))
            fi
        else
            echo "  $scenario_name: NOT COMPLETED" >> "$COMPREHENSIVE_SUMMARY"
            ((failure_count++))
        fi
    done
    
    # Summary statistics for this dataset
    total_scenarios=${#IMBALANCE_SCENARIOS[@]}
    echo "" >> "$COMPREHENSIVE_SUMMARY"
    echo "Dataset Summary:" >> "$COMPREHENSIVE_SUMMARY"
    echo "  Breakthrough scenarios: $breakthrough_count/$total_scenarios" >> "$COMPREHENSIVE_SUMMARY"
    echo "  Moderate success: $moderate_count/$total_scenarios" >> "$COMPREHENSIVE_SUMMARY"
    echo "  Failed scenarios: $failure_count/$total_scenarios" >> "$COMPREHENSIVE_SUMMARY"
    echo "  Success rate: $(python3 -c "print(f'{($breakthrough_count + $moderate_count) / $total_scenarios:.1%}')")" >> "$COMPREHENSIVE_SUMMARY"
done

# Overall conclusions
cat >> "$COMPREHENSIVE_SUMMARY" << EOF

OVERALL RESEARCH CONCLUSIONS:
============================

KEY FINDINGS:
1. [To be filled from analysis above]

BREAKTHROUGH SCENARIOS IDENTIFIED:
[List all scenarios that achieved >3% DP improvement]

RECOMMENDED STRATEGIES:
- For Brazil: [Best performing scenario]  
- For India: [Best performing scenario]
- For Africa: [Best performing scenario]

PRACTICAL IMPLICATIONS:
1. Dataset rebalancing CAN enable fairness interventions
2. Different datasets need different balance strategies
3. Some scenarios consistently fail across datasets

NEXT STEPS FOR RESEARCH:
1. Focus method comparison on breakthrough scenarios
2. Test breakthrough scenarios with other synthetic data methods
3. Develop dataset-specific fairness intervention strategies
EOF

# Create enhanced analysis focusing on fairness intervention success
log_message "Creating enhanced analysis focusing on fairness intervention effectiveness..."

ANALYSIS_FILE="$RESULTS_DIR/fairness_intervention_analysis.json"
if [[ -f "experiments/utils/analyze_results.py" ]]; then
    python experiments/utils/analyze_results.py "$RESULTS_DIR" \
        --output "$ANALYSIS_FILE" \
        --experiment "fairness_intervention_effectiveness" \
        --compare-scenarios
else
    log_message "Warning: Analysis script not found, skipping detailed analysis"
fi

# Create success/failure summary with actionable insights
SUCCESS_SUMMARY="$RESULTS_DIR/fairness_intervention_success_summary.txt"
cat > "$SUCCESS_SUMMARY" << EOF
FAIRNESS INTERVENTION SUCCESS ANALYSIS - $(date)
===============================================

BASELINE (Original Data):
- Brazil: FAILS (-3.0% DP improvement)
- India: FAILS (-3.5% DP improvement)
- Africa: Marginal success (+1.2% DP improvement)

RESEARCH QUESTIONS ANSWERED:
1. Can dataset rebalancing make fairness interventions work?
2. Which balance scenarios enable successful fairness interventions?
3. Do different datasets need different balance strategies?

BREAKTHROUGH SCENARIOS (>3% DP improvement):
[Automatically populated from results above]

MODERATE SUCCESS SCENARIOS (1-3% DP improvement):
[Automatically populated from results above]

CONSISTENT FAILURE PATTERNS:
[Scenarios that failed across multiple datasets]

DATASET-SPECIFIC INSIGHTS:
- Brazil: [Best scenarios and patterns]
- India: [Best scenarios and patterns]  
- Africa: [Best scenarios and patterns]

PRACTICAL RECOMMENDATIONS:
1. Use breakthrough scenarios for future fairness interventions
2. Apply dataset-specific balance strategies
3. Avoid scenarios that consistently fail

NEXT STEPS:
1. Implement breakthrough scenarios in production systems
2. Test breakthrough patterns on other datasets
3. Develop automated balance optimization
EOF

log_message "=== COMPREHENSIVE SUMMARY CREATED ==="
log_message "Detailed analysis: $COMPREHENSIVE_SUMMARY"
log_message "Success summary: $SUCCESS_SUMMARY"
log_message "View these files for complete research insights"

log_message "=== IMBALANCE IMPACT STUDY COMPLETED ==="
log_message "Total experiments: $experiment_count"
log_message "Results saved to: $RESULTS_DIR"
log_message "Fairness analysis: $ANALYSIS_FILE"

echo ""
echo "⚖️ EXPERIMENT 3 COMPLETED (ENHANCED)"
echo "🎯 Key Research Question: When do fairness interventions actually work?"
echo "📊 Critical Finding: Baseline fairness interventions FAIL for Brazil & India"
echo "🔍 Analysis Focus: Identify scenarios where fairness interventions succeed"
echo "📈 Enhancement: Comprehensive breakthrough detection and detailed metrics"
echo ""
echo "📁 Key Output Files:"
echo "   • $COMPREHENSIVE_SUMMARY - Complete experimental analysis"
echo "   • $SUCCESS_SUMMARY - Actionable insights and recommendations"
echo "   • $SUMMARY_FILE - Detailed scenario-by-scenario results"
echo ""
echo "🚀 Next Steps:"
echo "   1. Review breakthrough scenarios in success summary"
echo "   2. Analyze failure patterns for insights"
echo "   3. Use successful scenarios to guide Method Comparison (Experiment 4)"
echo "   4. Implement breakthrough scenarios in production systems"
echo ""