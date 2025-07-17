#!/bin/bash

# =============================================================================
# EXPERIMENT 3: LLM-BASED IMBALANCE IMPACT STUDY
# Research question: How do sensitive+label distribution combinations affect 
# fairness interventions when using LLM-based synthetic data generation?
#
# This script focuses exclusively on LLM-based synthetic data generation
# to understand its effectiveness across different distribution scenarios.
#
# DISTRIBUTION LOGIC CLARIFICATION:
# - sensitive_ratio = proportion of MALES (privileged group), remainder = FEMALES
# - label_ratio = proportion of DROPOUT cases (positive class), remainder = NO DROPOUT
# - All distributions sum to 100% as expected
# =============================================================================

set -e

# Configuration
DATASETS=("brazil" "india" "africa")
METHOD="llm_async"  # Focus only on LLM method
EXPERIMENT_NAME="03_imbalance_impact_llm"
RESULTS_DIR="./experiments/results/$EXPERIMENT_NAME"
LOG_FILE="$RESULTS_DIR/experiment.log"
API_KEY="sk-8d71fd82b1e34ce48e31718a1f3647bf"

# --- Create directories ---
mkdir -p "$RESULTS_DIR"

# --- Logging function ---
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_message "=== LLM-BASED IMBALANCE IMPACT STUDY ==="
log_message "METHOD: LLM-based synthetic data generation only"
log_message "FOCUS: Understanding LLM performance across distribution scenarios"
log_message "ENHANCEMENT: Robust timeout monitoring and progress tracking"

# --- Scenario Definitions ---
declare -A DISTRIBUTION_SCENARIOS=(
    # PERFECT BALANCE (Baseline Success Case)
    ["balanced_balanced"]="0.5 0.5"                        # 50%M+50%F, 50%D+50%ND
    
    # SINGLE DIMENSION IMBALANCES (Test one factor at a time)
    ["moderate_sensitive_balanced"]="0.65 0.5"             # 65%M+35%F, 50%D+50%ND
    ["extreme_sensitive_balanced"]="0.8 0.5"               # 80%M+20%F, 50%D+50%ND
    ["balanced_moderate_labels"]="0.5 0.7"                 # 50%M+50%F, 70%D+30%ND
    ["balanced_extreme_labels"]="0.5 0.8"                  # 50%M+50%F, 80%D+20%ND
    
    # COMPOUND IMBALANCES (Test interaction effects)
    ["moderate_compound"]="0.6 0.6"                        # 60%M+40%F, 60%D+40%ND
    ["extreme_compound"]="0.7 0.7"                         # 70%M+30%F, 70%D+30%ND
    
    # REVERSE/MINORITY BIAS (Test opposite direction)
    ["reverse_sensitive"]="0.3 0.5"                        # 30%M+70%F, 50%D+50%ND
    ["reverse_extreme_sensitive"]="0.2 0.5"                # 20%M+80%F, 50%D+50%ND
    
    # BASELINE COMPARISON
    ["original_ratios"]="original original"                 # Natural dataset ratios (no augmentation)
)

# --- Summary and Report Setup ---
STUDY_SUMMARY="$RESULTS_DIR/llm_study_summary.txt"
cat > "$STUDY_SUMMARY" << EOF
LLM-BASED IMBALANCE IMPACT STUDY - $(date)
==========================================

RESEARCH DESIGN OVERVIEW:
This experiment systematically explores LLM-based synthetic data generation across:
1. Sensitive attribute distribution (Gender: Male vs Female ratio)
2. Label distribution (Dropout: Yes vs No ratio)  

DISTRIBUTION SPECIFICATION LOGIC:
=================================
For binary attributes, ratios specify the proportion of the PRIVILEGED/POSITIVE class:

SENSITIVE ATTRIBUTES (Gender):
- sensitive_ratio = proportion of MALES (privileged group)
- Remaining proportion = FEMALES (unprivileged group)
- Example: 0.6 = 60% Male + 40% Female = 100% ✓

LABELS (Dropout):
- label_ratio = proportion of DROPOUT cases (positive class)
- Remaining proportion = NO DROPOUT cases (negative class)  
- Example: 0.7 = 70% Dropout + 30% No Dropout = 100% ✓

SYSTEMATIC SCENARIO MATRIX (All distributions verified to sum to 100%):
=====================================================================

EOF

# Add detailed scenario descriptions with explicit percentage breakdowns
for scenario_name in "${!DISTRIBUTION_SCENARIOS[@]}"; do
    scenario_value="${DISTRIBUTION_SCENARIOS[$scenario_name]}"
    
    if [[ "$scenario_name" == "original_ratios" ]]; then
        echo "- $scenario_name: Natural dataset distributions (no augmentation)" >> "$STUDY_SUMMARY"
    else
        IFS=' ' read -r sensitive_ratio label_ratio <<< "$scenario_value"
        sensitive_pct=$(python3 -c "print(f'{float('$sensitive_ratio')*100:.0f}')")
        sensitive_comp_pct=$(python3 -c "print(f'{(1-float('$sensitive_ratio'))*100:.0f}')")
        label_pct=$(python3 -c "print(f'{float('$label_ratio')*100:.0f}')")
        label_comp_pct=$(python3 -c "print(f'{(1-float('$label_ratio'))*100:.0f}')")
        
        echo "- $scenario_name ($sensitive_ratio, $label_ratio): ${sensitive_pct}% Male + ${sensitive_comp_pct}% Female, ${label_pct}% Dropout + ${label_comp_pct}% No Dropout" >> "$STUDY_SUMMARY"
    fi
done

cat >> "$STUDY_SUMMARY" << EOF

EXPERIMENTAL DESIGN RATIONALE:
==============================
- PERFECT BALANCE: Tests ideal fairness scenario (balanced_balanced)
- SINGLE DIMENSION IMBALANCES: Isolates impact of one imbalanced factor
- COMPOUND IMBALANCES: Tests interaction effects of multiple imbalances
- REVERSE BIAS TESTING: Tests scenarios where the unprivileged group is the majority
- BASELINE CONTROL: Uses natural dataset distributions without augmentation

EXPERIMENTAL MATRIX:
===================
- Datasets: 3 (Brazil, India, Africa)
- Scenarios: ${#DISTRIBUTION_SCENARIOS[@]} distribution combinations
- Method: LLM-based synthetic data generation
- Total experiments: $((${#DATASETS[@]} * ${#DISTRIBUTION_SCENARIOS[@]}))

SUCCESS CRITERIA:
- BREAKTHROUGH: >3% DP improvement with <5% accuracy cost
- MODERATE: 1-3% DP improvement
- FAILURE: ≤0% DP improvement or >5% accuracy cost

TIMEOUT HANDLING:
- LLM calls have 10-minute timeout with 30-second progress monitoring
- Robust error handling and process management

DETAILED EXPERIMENTAL RESULTS:
==============================
EOF

total_experiments=$((${#DATASETS[@]} * ${#DISTRIBUTION_SCENARIOS[@]}))
log_message "Planning $total_experiments LLM-based experiments across validated distribution scenarios"

experiment_count=0

# --- CSV Headers ---
RESULTS_MATRIX="$RESULTS_DIR/llm_results_matrix.csv"
echo "Dataset,Scenario,Sensitive_Ratio,Label_Ratio,DP_Improvement,Accuracy_Cost,Category,Final_DP_Diff,Balance_Achieved,Status,Duration_Seconds" > "$RESULTS_MATRIX"

PERFORMANCE_SUMMARY="$RESULTS_DIR/llm_performance_summary.csv"
echo "Dataset,Scenario,DP_Improvement,Category,Success_Rate,Avg_Duration" > "$PERFORMANCE_SUMMARY"

# --- Main Experiment Loop ---
for dataset in "${DATASETS[@]}"; do
    log_message "Starting LLM-based study for $dataset..."
    
    dataset_results_dir="$RESULTS_DIR/${dataset}_llm"
    mkdir -p "$dataset_results_dir"
    
    echo "" >> "$STUDY_SUMMARY"
    echo "=================================================================" >> "$STUDY_SUMMARY"
    echo "=== $dataset LLM-BASED DISTRIBUTION ANALYSIS ===" >> "$STUDY_SUMMARY"
    echo "=================================================================" >> "$STUDY_SUMMARY"
    
    for scenario_name in "${!DISTRIBUTION_SCENARIOS[@]}"; do
        scenario_value="${DISTRIBUTION_SCENARIOS[$scenario_name]}"
        
        echo "" >> "$STUDY_SUMMARY"
        echo "SCENARIO: $scenario_name" >> "$STUDY_SUMMARY"
        echo "----------------------------------------" >> "$STUDY_SUMMARY"
        
        if [[ "$scenario_name" == "original_ratios" ]]; then
            sensitive_ratio="original"
            label_ratio="original"
        else
            IFS=' ' read -r sensitive_ratio label_ratio <<< "$scenario_value"
        fi
        
        log_message "[$((++experiment_count))/$total_experiments] $dataset - $scenario_name - LLM"
        start_time=$(date +%s)
        
        output_file="$dataset_results_dir/${scenario_name}_llm.txt"
        
        # --- CORE EXECUTION BLOCK ---
        # Build command into an array for robustness
        cmd=() 
        if [[ "$scenario_name" == "original_ratios" ]]; then
            cmd=(python main.py "yaml/${dataset}.yaml" \
                --fairness \
                --save_results \
                --results_folder "$dataset_results_dir" \
                --scenario_name "original_llm" \
                --api_key "$API_KEY")
        else
            cmd=(python main.py "yaml/${dataset}.yaml" \
                --balanced \
                --sensitive_ratio "$sensitive_ratio" \
                --label_ratio "$label_ratio" \
                --fairness \
                --save_results \
                --results_folder "$dataset_results_dir" \
                --scenario_name "${scenario_name}_llm" \
                --method "$METHOD" \
                --api_key "$API_KEY")
        fi

        # --- LLM Execution with Timeout & Monitoring ---
        experiment_status=0
        log_message "⏱️  Starting LLM API call (timeout: 10 minutes)..."
        
        # Run LLM in background with timeout and monitoring
        "${cmd[@]}" < /dev/null > "$output_file" 2>&1 &
        python_pid=$!
        
        monitor_count=0
        # 10 minute timeout with 30s checks = 20 iterations
        while kill -0 $python_pid 2>/dev/null; do
            sleep 30
            ((monitor_count++))
            log_message "⏳ LLM call in progress... ${monitor_count}x30s elapsed"
            
            if [[ $monitor_count -ge 20 ]]; then
                log_message "⏰ TIMEOUT: Killing LLM call - exceeded 10 minutes"
                kill $python_pid 2>/dev/null
                sleep 1 # Give it a moment to terminate gracefully
                kill -9 $python_pid 2>/dev/null # Force kill if needed
                break
            fi
        done
        
        wait $python_pid 2>/dev/null
        experiment_status=$?
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # --- Metric Extraction & Analysis ---
        status="ERROR"
        dp_improvement="N/A"
        accuracy_cost="N/A"
        final_dp_diff="N/A"
        balance_achieved="N/A"
        category="ERROR"

        if [[ $experiment_status -eq 143 || $experiment_status -eq 137 ]]; then
            status="TIMEOUT"
            category="TIMEOUT"
            echo "LLM: TIMEOUT (>10 minutes)" >> "$STUDY_SUMMARY"
            log_message "⏰ TIMEOUT: $dataset-$scenario_name-LLM exceeded 10 minutes (${duration}s)"
        elif [[ $experiment_status -eq 0 && -f "$output_file" && -s "$output_file" ]]; then
            status="SUCCESS"
            dp_improvement="0"
            accuracy_cost="0"
            final_dp_diff="0"
            balance_achieved="No"
            category="FAILED"
            
            if grep -q "DP improvement" "$output_file"; then
                dp_improvement=$(grep "DP improvement" "$output_file" | tail -1 | awk '{print $NF}')
            fi
            if grep -q "Accuracy cost" "$output_file"; then
                accuracy_cost=$(grep "Accuracy cost" "$output_file" | tail -1 | awk '{print $NF}')
            fi
            if grep -q "Demographic Parity Difference" "$output_file"; then
                final_dp_diff=$(grep "Demographic Parity Difference" "$output_file" | tail -1 | awk '{print $NF}')
            fi
            if grep -q "Balance Achievement" "$output_file"; then
                balance_achieved="Yes"
            fi
            
            if python3 -c "import sys; exit(0 if float('$dp_improvement') > 0.03 and abs(float('$accuracy_cost')) < 0.05 else 1)" 2>/dev/null; then
                category="BREAKTHROUGH"
            elif python3 -c "import sys; exit(0 if 0.01 <= float('$dp_improvement') <= 0.03 else 1)" 2>/dev/null; then
                category="MODERATE"
            elif python3 -c "import sys; exit(0 if abs(float('$accuracy_cost')) > 0.05 else 1)" 2>/dev/null; then
                category="HIGH_ACCURACY_COST"
            elif python3 -c "import sys; exit(0 if float('$dp_improvement') <= 0 else 1)" 2>/dev/null; then
                category="FAILED"
            else
                category="UNCLEAR"
            fi
            
            echo "LLM Results (${duration}s):" >> "$STUDY_SUMMARY"
            echo "  DP Improvement: $dp_improvement ($category)" >> "$STUDY_SUMMARY"
            echo "  Accuracy Cost: $accuracy_cost" >> "$STUDY_SUMMARY"
            echo "  Final DP Difference: $final_dp_diff" >> "$STUDY_SUMMARY"
            echo "  Balance Achieved: $balance_achieved" >> "$STUDY_SUMMARY"
            
            if [[ "$category" == "BREAKTHROUGH" ]]; then
                echo "  🎉 BREAKTHROUGH ACHIEVED!" >> "$STUDY_SUMMARY"
            elif [[ "$category" == "MODERATE" ]]; then
                echo "  ✅ MODERATE SUCCESS" >> "$STUDY_SUMMARY"
            fi
            
            log_message "✅ SUCCESS: $dataset-$scenario_name-LLM: $category (+$dp_improvement DP, $accuracy_cost AC, ${duration}s)"

        else
            echo "LLM: ERROR (exit code: $experiment_status, ${duration}s)" >> "$STUDY_SUMMARY"
            if [[ -f "$output_file" ]]; then
                echo "  Error details: $(tail -3 "$output_file" | head -1)" >> "$STUDY_SUMMARY"
            fi
            log_message "❌ ERROR: $dataset-$scenario_name-LLM failed (exit code: $experiment_status, ${duration}s)"
        fi
        
        # Add to comprehensive results matrix
        echo "$dataset,$scenario_name,$sensitive_ratio,$label_ratio,$dp_improvement,$accuracy_cost,$category,$final_dp_diff,$balance_achieved,$status,$duration" >> "$RESULTS_MATRIX"
        
        # Pause to avoid overwhelming API
        sleep 5
    done
    
    log_message "Completed all distribution scenarios for $dataset"
done

# --- Final Analysis & Reporting ---
log_message "Creating final LLM-based analysis reports..."

COMPREHENSIVE_ANALYSIS="$RESULTS_DIR/llm_analysis_and_recommendations.txt"
cat > "$COMPREHENSIVE_ANALYSIS" << EOF
LLM-BASED SYNTHETIC DATA GENERATION ANALYSIS - $(date)
======================================================

EXECUTIVE SUMMARY:
This study evaluated LLM-based synthetic data generation across various
distribution scenarios to understand its effectiveness for fairness interventions.

METHODOLOGY:
- Datasets: Brazil, India, Africa education datasets
- Scenarios: ${#DISTRIBUTION_SCENARIOS[@]} different distribution combinations
- Method: LLM-based synthetic data generation with API calls
- Total Experiments: $total_experiments

KEY FINDINGS:
=============
EOF

# Generate summary statistics
python3 << 'EOF'
import csv
import pandas as pd
from collections import defaultdict

# Read results
results_file = "$(echo $RESULTS_MATRIX)"
try:
    df = pd.read_csv(results_file)
    
    # Overall statistics
    total_experiments = len(df)
    successful_experiments = len(df[df['Status'] == 'SUCCESS'])
    breakthrough_count = len(df[df['Category'] == 'BREAKTHROUGH'])
    moderate_count = len(df[df['Category'] == 'MODERATE'])
    failed_count = len(df[df['Category'] == 'FAILED'])
    timeout_count = len(df[df['Category'] == 'TIMEOUT'])
    
    print(f"OVERALL PERFORMANCE:")
    print(f"- Total Experiments: {total_experiments}")
    print(f"- Successful Runs: {successful_experiments} ({successful_experiments/total_experiments*100:.1f}%)")
    print(f"- Breakthrough Results: {breakthrough_count} ({breakthrough_count/total_experiments*100:.1f}%)")
    print(f"- Moderate Success: {moderate_count} ({moderate_count/total_experiments*100:.1f}%)")
    print(f"- Failed: {failed_count} ({failed_count/total_experiments*100:.1f}%)")
    print(f"- Timeouts: {timeout_count} ({timeout_count/total_experiments*100:.1f}%)")
    print()
    
    # Best performing scenarios
    successful_df = df[df['Status'] == 'SUCCESS']
    if len(successful_df) > 0:
        # Convert DP_Improvement to numeric, handling 'N/A' values
        successful_df = successful_df.copy()
        successful_df['DP_Numeric'] = pd.to_numeric(successful_df['DP_Improvement'], errors='coerce')
        
        best_scenarios = successful_df.nlargest(5, 'DP_Numeric')[['Dataset', 'Scenario', 'DP_Improvement', 'Category']]
        print("TOP 5 PERFORMING SCENARIOS:")
        for _, row in best_scenarios.iterrows():
            print(f"- {row['Dataset']}/{row['Scenario']}: +{row['DP_Improvement']} DP ({row['Category']})")
        print()
    
    # Dataset performance
    print("DATASET-SPECIFIC PERFORMANCE:")
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        dataset_success = len(dataset_df[dataset_df['Status'] == 'SUCCESS'])
        dataset_breakthrough = len(dataset_df[dataset_df['Category'] == 'BREAKTHROUGH'])
        print(f"- {dataset}: {dataset_success}/{len(dataset_df)} successful, {dataset_breakthrough} breakthroughs")
    print()
    
except Exception as e:
    print(f"Error analyzing results: {e}")
    print("Manual analysis required - check CSV files directly.")
EOF >> "$COMPREHENSIVE_ANALYSIS"

echo "" >> "$COMPREHENSIVE_ANALYSIS"
echo "RECOMMENDATIONS:" >> "$COMPREHENSIVE_ANALYSIS"
echo "================" >> "$COMPREHENSIVE_ANALYSIS"
echo "1. Focus on scenarios that achieved BREAKTHROUGH results" >> "$COMPREHENSIVE_ANALYSIS"
echo "2. Investigate timeout scenarios - may need API optimization" >> "$COMPREHENSIVE_ANALYSIS"
echo "3. Consider computational cost vs. fairness improvement trade-offs" >> "$COMPREHENSIVE_ANALYSIS"
echo "4. Validate top-performing scenarios with additional metrics" >> "$COMPREHENSIVE_ANALYSIS"
echo "" >> "$COMPREHENSIVE_ANALYSIS"
echo "For detailed results, see:" >> "$COMPREHENSIVE_ANALYSIS"
echo "- LLM Results Matrix: llm_results_matrix.csv" >> "$COMPREHENSIVE_ANALYSIS"
echo "- Detailed Summary: llm_study_summary.txt" >> "$COMPREHENSIVE_ANALYSIS"

log_message "=== LLM-BASED STUDY COMPLETED ==="
log_message "Results matrix: $RESULTS_MATRIX"
log_message "Detailed summary: $STUDY_SUMMARY"
log_message "Analysis report: $COMPREHENSIVE_ANALYSIS"

echo ""
echo "🎯 LLM-BASED IMBALANCE IMPACT STUDY COMPLETED"
echo "🤖 Focus: LLM synthetic data generation across distribution scenarios"
echo "⏱️ Enhanced with robust timeout monitoring for API stability"
echo "📋 Evidence: $total_experiments LLM-based experiments executed systematically"
echo ""
echo "📁 Key Deliverables in '$RESULTS_DIR':"
echo "   📊 LLM Results Matrix: llm_results_matrix.csv"
echo "   📝 Detailed Study Summary: llm_study_summary.txt"
echo "   📈 Analysis & Recommendations: llm_analysis_and_recommendations.txt"
echo ""
echo "🎉 READY FOR LLM PERFORMANCE ANALYSIS!"
echo "Use the generated files to understand LLM effectiveness across scenarios."
echo ""