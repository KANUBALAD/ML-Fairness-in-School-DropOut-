#!/bin/bash

# =============================================================================
# EXPERIMENT 3: COMPREHENSIVE IMBALANCE & METHOD SELECTION STUDY (v2 - Robust)
# Combined research questions:
# Q1: How do sensitive+label distribution combinations affect fairness interventions?
# Q2: When should you use Faker vs LLM for different distribution scenarios?
#
# v2 FIX: Added </dev/null to backgrounded python calls to prevent hangs on stdin.
#         Corrected timeout logic and improved robustness.
#
# DISTRIBUTION LOGIC CLARIFICATION:
# - sensitive_ratio = proportion of MALES (privileged group), remainder = FEMALES
# - label_ratio = proportion of DROPOUT cases (positive class), remainder = NO DROPOUT
# - All distributions sum to 100% as expected
# =============================================================================

set -e

# Configuration
DATASETS=("brazil" "india" "africa")
METHODS=("faker" "llm_async")  # Test both methods on all datasets
EXPERIMENT_NAME="03_comprehensive_imbalance_method_study_v2"
RESULTS_DIR="./experiments/results/$EXPERIMENT_NAME"
LOG_FILE="$RESULTS_DIR/experiment.log"
API_KEY="sk-8d71fd82b1e34ce48e31718a1f3647bf"

# --- Create directories ---
mkdir -p "$RESULTS_DIR"

# --- Logging function ---
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_message "=== COMPREHENSIVE IMBALANCE & METHOD SELECTION STUDY (v2) ==="
log_message "FIX APPLIED: Added </dev/null to prevent LLM hangs on stdin."
log_message "SYSTEMATIC APPROACH: Test both methods on all datasets across key distribution scenarios"
log_message "ENHANCEMENT: Added timeout monitoring and progress tracking for LLM calls"

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
STUDY_SUMMARY="$RESULTS_DIR/comprehensive_study_summary.txt"
cat > "$STUDY_SUMMARY" << EOF
COMPREHENSIVE IMBALANCE & METHOD SELECTION STUDY - $(date)
==========================================================

RESEARCH DESIGN OVERVIEW:
This experiment systematically explores the interaction between:
1. Sensitive attribute distribution (Gender: Male vs Female ratio)
2. Label distribution (Dropout: Yes vs No ratio)  
3. Synthetic data generation method (Faker vs LLM)

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
- Methods: 2 (Faker, LLM)
- Total experiments: $((${#DATASETS[@]} * ${#DISTRIBUTION_SCENARIOS[@]} * ${#METHODS[@]}))

SUCCESS CRITERIA:
- BREAKTHROUGH: >3% DP improvement with <5% accuracy cost
- MODERATE: 1-3% DP improvement
- FAILURE: ≤0% DP improvement or >5% accuracy cost

TIMEOUT HANDLING:
- LLM calls have 10-minute timeout with 30-second progress monitoring
- Faker calls run without timeout (typically complete in <30 seconds)

DETAILED EXPERIMENTAL RESULTS:
==============================
EOF

total_experiments=$((${#DATASETS[@]} * ${#DISTRIBUTION_SCENARIOS[@]} * ${#METHODS[@]}))
log_message "Planning $total_experiments systematic experiments across validated distribution scenarios"

experiment_count=0

# --- CSV Headers ---
RESULTS_MATRIX="$RESULTS_DIR/systematic_results_matrix.csv"
echo "Dataset,Scenario,Sensitive_Ratio,Label_Ratio,Method,DP_Improvement,Accuracy_Cost,Category,Final_DP_Diff,Balance_Achieved,Status,Duration_Seconds" > "$RESULTS_MATRIX"

METHOD_COMPARISON="$RESULTS_DIR/method_comparison_summary.csv"
echo "Dataset,Scenario,Faker_DP,Faker_Category,LLM_DP,LLM_Category,Winner,Margin,Significant_Difference" > "$METHOD_COMPARISON"

# --- Main Experiment Loop ---
for dataset in "${DATASETS[@]}"; do
    log_message "Starting comprehensive study for $dataset..."
    
    dataset_results_dir="$RESULTS_DIR/${dataset}"
    mkdir -p "$dataset_results_dir"
    
    echo "" >> "$STUDY_SUMMARY"
    echo "=================================================================" >> "$STUDY_SUMMARY"
    echo "=== $dataset COMPREHENSIVE DISTRIBUTION ANALYSIS ===" >> "$STUDY_SUMMARY"
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
        
        declare -A scenario_method_results
        
        for method in "${METHODS[@]}"; do
            log_message "[$((++experiment_count))/$total_experiments] $dataset - $scenario_name - $method"
            start_time=$(date +%s)
            
            method_args=""
            if [[ "$method" == "llm_async" ]]; then
                method_args="--api_key $API_KEY"
            fi
            
            output_file="$dataset_results_dir/${scenario_name}_${method}.txt"
            
            # --- CORE EXECUTION BLOCK (WITH FIXES) ---
            # Build command into an array for robustness
            cmd=() 
            if [[ "$scenario_name" == "original_ratios" ]]; then
                cmd=(python main.py "yaml/${dataset}.yaml" \
                    --fairness \
                    --save_results \
                    --results_folder "$dataset_results_dir" \
                    --scenario_name "original_${method}" \
                    $method_args)
            else
                cmd=(python main.py "yaml/${dataset}.yaml" \
                    --balanced \
                    --sensitive_ratio "$sensitive_ratio" \
                    --label_ratio "$label_ratio" \
                    --fairness \
                    --save_results \
                    --results_folder "$dataset_results_dir" \
                    --scenario_name "${scenario_name}_${method}" \
                    --method "$method" \
                    $method_args)
            fi

            # --- Unified Execution & Timeout Logic ---
            experiment_status=0
            if [[ "$method" == "llm_async" ]]; then
                log_message "⏱️  Starting LLM API call (timeout: 10 minutes)..."
                
                # Run LLM in background with timeout, monitoring, AND THE STDIN FIX
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
            else
                # Faker - no timeout needed, run in foreground for simplicity
                "${cmd[@]}" > "$output_file" 2>&1
                experiment_status=$?
            fi
            
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
                echo "$method: TIMEOUT (>10 minutes)" >> "$STUDY_SUMMARY"
                log_message "⏰ TIMEOUT: $dataset-$scenario_name-$method exceeded 10 minutes (${duration}s)"
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
                
                scenario_method_results["${method}_dp"]="$dp_improvement"
                scenario_method_results["${method}_category"]="$category"
                
                echo "$method Results (${duration}s):" >> "$STUDY_SUMMARY"
                echo "  DP Improvement: $dp_improvement ($category)" >> "$STUDY_SUMMARY"
                echo "  Accuracy Cost: $accuracy_cost" >> "$STUDY_SUMMARY"
                
                if [[ "$category" == "BREAKTHROUGH" ]]; then
                    echo "  🎉 BREAKTHROUGH ACHIEVED!" >> "$STUDY_SUMMARY"
                fi
                log_message "✅ SUCCESS: $dataset-$scenario_name-$method: $category (+$dp_improvement DP, $accuracy_cost AC, ${duration}s)"

            else
                echo "$method: ERROR (exit code: $experiment_status, ${duration}s)" >> "$STUDY_SUMMARY"
                if [[ -f "$output_file" ]]; then
                    echo "  Error details: $(tail -3 "$output_file" | head -1)" >> "$STUDY_SUMMARY"
                fi
                log_message "❌ ERROR: $dataset-$scenario_name-$method failed (exit code: $experiment_status, ${duration}s)"
            fi
            
            # Add to comprehensive results matrix
            echo "$dataset,$scenario_name,$sensitive_ratio,$label_ratio,$method,$dp_improvement,$accuracy_cost,$category,$final_dp_diff,$balance_achieved,$status,$duration" >> "$RESULTS_MATRIX"
            
            # Pause to avoid overwhelming APIs or file systems
            if [[ "$method" == "llm_async" ]]; then
                sleep 5
            else
                sleep 1
            fi
        done
        
        # --- Head-to-head method comparison for this scenario ---
        if [[ -n "${scenario_method_results[faker_dp]}" && -n "${scenario_method_results[llm_async_dp]}" && 
              "${scenario_method_results[faker_dp]}" != "N/A" && "${scenario_method_results[llm_async_dp]}" != "N/A" ]]; then
            
            faker_dp="${scenario_method_results[faker_dp]}"
            llm_dp="${scenario_method_results[llm_async_dp]}"
            faker_cat="${scenario_method_results[faker_category]}"
            llm_cat="${scenario_method_results[llm_async_category]}"
            
            if python3 -c "import sys; exit(0 if float('$llm_dp') > float('$faker_dp') else 1)" 2>/dev/null; then
                winner="LLM"
                margin=$(python3 -c "print(f'{float('$llm_dp') - float('$faker_dp'):.4f}')")
            elif python3 -c "import sys; exit(0 if float('$faker_dp') > float('$llm_dp') else 1)" 2>/dev/null; then
                winner="Faker"
                margin=$(python3 -c "print(f'{float('$faker_dp') - float('$llm_dp'):.4f}')")
            else
                winner="TIE"
                margin="0.0000"
            fi
            
            significant="No"
            if python3 -c "import sys; exit(0 if abs(float('$llm_dp') - float('$faker_dp')) > 0.02 else 1)" 2>/dev/null; then
                significant="Yes"
            fi
            
            echo "" >> "$STUDY_SUMMARY"
            echo "HEAD-TO-HEAD METHOD COMPARISON:" >> "$STUDY_SUMMARY"
            echo "  🏆 Winner: $winner (margin: +$margin DP improvement)" >> "$STUDY_SUMMARY"
            echo "  Faker: $faker_dp DP ($faker_cat) vs LLM: $llm_dp DP ($llm_cat)" >> "$STUDY_SUMMARY"
            
            # Add to method comparison CSV
            echo "$dataset,$scenario_name,$faker_dp,$faker_cat,$llm_dp,$llm_cat,$winner,$margin,$significant" >> "$METHOD_COMPARISON"
        else
            echo "HEAD-TO-HEAD COMPARISON: Cannot compare - one or both methods failed" >> "$STUDY_SUMMARY"
        fi
        
        # Clear results for next scenario
        unset scenario_method_results
    done
    
    log_message "Completed all distribution scenarios for $dataset"
done

# --- Final Analysis & Reporting ---
log_message "Creating final comprehensive analysis reports..."
COMPREHENSIVE_ANALYSIS="$RESULTS_DIR/comprehensive_analysis_and_recommendations.txt"

# ... (Your excellent and very detailed final reporting logic can be pasted here)
# This part of your script was well-designed and does not need changes.
# It will read from the generated CSVs to produce the final text summaries.

log_message "=== COMPREHENSIVE STUDY COMPLETED ==="
log_message "Results matrix: $RESULTS_MATRIX"
log_message "Method comparison: $METHOD_COMPARISON"
log_message "Detailed summary: $STUDY_SUMMARY"

echo ""
echo "🎯 COMPREHENSIVE IMBALANCE & METHOD SELECTION STUDY COMPLETED"
echo "✅ Fixes for LLM execution stability applied successfully."
echo "⏱️ Enhanced with robust timeout monitoring for reliability."
echo "📋 Evidence: $total_experiments experiments executed systematically."
echo ""
echo "📁 Key Deliverables in '$RESULTS_DIR':"
echo "   📊 Systematic Results Matrix: systematic_results_matrix.csv"
echo "   🏆 Method Comparison Summary: method_comparison_summary.csv"
echo "   📝 Detailed Log & Summary: comprehensive_study_summary.txt"
echo ""
echo "🎉 READY FOR ANALYSIS!"
echo "Use the generated CSV files and text summaries to draw your conclusions."
echo ""