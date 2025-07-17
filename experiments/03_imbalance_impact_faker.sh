#!/bin/bash

# =============================================================================
# EXPERIMENT 3: COMPREHENSIVE IMBALANCE IMPACT ANALYSIS WITH CSV SAVING
# =============================================================================

set -e

# Configuration
DATASETS=("brazil" "india" "africa")
METHODS=("faker")
EXPERIMENT_NAME="03_comprehensive_imbalance_faker_with_csvs"
RESULTS_DIR="./experiments/results/$EXPERIMENT_NAME"
LOG_FILE="$RESULTS_DIR/experiment.log"

# Create main results directory
mkdir -p "$RESULTS_DIR"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_message "=== COMPREHENSIVE IMBALANCE IMPACT ANALYSIS WITH CSV SAVING ==="
log_message "Augmented datasets will be automatically saved for each experiment"

# Scenario definitions
declare -A DISTRIBUTION_SCENARIOS=(
    ["balanced_balanced"]="0.5 0.5"
    ["moderate_sensitive_balanced"]="0.65 0.5"
    ["extreme_sensitive_balanced"]="0.8 0.5"
    ["balanced_moderate_labels"]="0.5 0.7"
    ["balanced_extreme_labels"]="0.5 0.8"
    ["moderate_compound"]="0.6 0.6"
    ["extreme_compound"]="0.7 0.7"
    ["reverse_sensitive"]="0.3 0.5"
    ["reverse_extreme_sensitive"]="0.2 0.5"
    ["original_ratios"]="original original"
)

# CSV Headers for results matrix
RESULTS_MATRIX="$RESULTS_DIR/faker_results_matrix.csv"
echo "Dataset,Scenario,Sensitive_Ratio,Label_Ratio,Method,DP_Improvement,Accuracy_Cost,Category,Final_DP_Diff,Balance_Achieved,Status,Duration_Seconds,CSV_Path" > "$RESULTS_MATRIX"

# Create CSV tracking file for all generated datasets
CSV_INVENTORY="$RESULTS_DIR/generated_csv_inventory.csv"
echo "Dataset,Scenario,Method,CSV_Path,Dataset_Size,Original_Size,Augmentation_Count,Timestamp" > "$CSV_INVENTORY"

# Enhanced categorization function
categorize_result() {
    local dp_val="$1"
    local acc_val="$2"
    
    # Validate inputs are numbers
    if ! [[ "$dp_val" =~ ^-?[0-9]*\.?[0-9]+$ ]] || ! [[ "$acc_val" =~ ^-?[0-9]*\.?[0-9]+$ ]]; then
        echo "INPUT_ERROR"
        return
    fi
    
    # Use Python for reliable float comparison
    python3 -c "
import sys
try:
    dp = float('$dp_val')
    ac = abs(float('$acc_val'))
    
    # Categorization logic
    if dp > 0.03 and ac < 0.05:
        print('BREAKTHROUGH')
    elif dp > 0.03 and ac >= 0.05:
        print('HIGH_ACCURACY_COST') 
    elif 0.01 <= dp <= 0.03:
        print('MODERATE')
    elif 0 < dp < 0.01:
        print('UNCLEAR')
    elif dp <= 0:
        print('FAILED')
    else:
        print('LOGIC_ERROR')
except Exception as e:
    print('PYTHON_ERROR')
"
}

# Enhanced metric extraction function
extract_metrics() {
    local output_file="$1"
    local dp_improvement="0"
    local accuracy_cost="0"
    local final_dp_diff="N/A"
    local balance_achieved="No"
    local csv_path="N/A"
    
    if [[ ! -f "$output_file" || ! -s "$output_file" ]]; then
        echo "0,0,N/A,No,N/A"
        return
    fi
    
    # Enhanced DP improvement extraction
    if grep -q "DP improvement:" "$output_file"; then
        dp_improvement=$(grep "DP improvement:" "$output_file" | tail -1 | sed 's/.*DP improvement: *//' | sed 's/[^0-9.-].*//' | head -1)
    elif grep -q "Demographic Parity improvement:" "$output_file"; then
        dp_improvement=$(grep "Demographic Parity improvement:" "$output_file" | tail -1 | sed 's/.*improvement: *//' | sed 's/[^0-9.-].*//' | head -1)
    fi
    
    # Enhanced accuracy cost extraction
    if grep -q "Accuracy cost:" "$output_file"; then
        accuracy_cost=$(grep "Accuracy cost:" "$output_file" | tail -1 | sed 's/.*cost: *//' | sed 's/[^0-9.-].*//' | head -1)
    elif grep -q "Accuracy change:" "$output_file"; then
        accuracy_cost=$(grep "Accuracy change:" "$output_file" | tail -1 | sed 's/.*change: *//' | sed 's/[^0-9.-].*//' | head -1)
    fi
    
    # Extract CSV path from the output
    if grep -q "Saved augmented dataset to:" "$output_file"; then
        csv_path=$(grep "Saved augmented dataset to:" "$output_file" | tail -1 | sed 's/.*Saved augmented dataset to: *//' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
    fi
    
    # Enhanced final DP difference extraction
    if grep -q "Final.*DP.*[Dd]ifference" "$output_file"; then
        final_dp_diff=$(grep "Final.*DP.*[Dd]ifference" "$output_file" | tail -1 | sed 's/.*[Dd]ifference[: ]*//' | sed 's/[^0-9.-].*//' | head -1)
    fi
    
    # Balance achievement check
    if grep -qE "(Balance.*achieved|Successfully.*balanced|Rebalancing.*successful)" "$output_file"; then
        balance_achieved="Yes"
    fi
    
    # Validate extracted values
    [[ ! "$dp_improvement" =~ ^-?[0-9]*\.?[0-9]+$ ]] && dp_improvement="0"
    [[ ! "$accuracy_cost" =~ ^-?[0-9]*\.?[0-9]+$ ]] && accuracy_cost="0"
    [[ ! "$final_dp_diff" =~ ^-?[0-9]*\.?[0-9]+$ ]] && final_dp_diff="N/A"
    
    echo "$dp_improvement,$accuracy_cost,$final_dp_diff,$balance_achieved,$csv_path"
}

# Function to track generated CSV files
track_generated_csv() {
    local dataset="$1"
    local scenario="$2"
    local method="$3"
    local csv_path="$4"
    local output_file="$5"
    
    if [[ "$csv_path" != "N/A" && -f "$csv_path" ]]; then
        # Get dataset size information
        local dataset_size=$(wc -l < "$csv_path" 2>/dev/null || echo "0")
        local original_size="0"
        local augmentation_count="0"
        
        # Extract original and final sizes from output
        if [[ -f "$output_file" ]]; then
            if grep -q "Original size:" "$output_file"; then
                original_size=$(grep "Original size:" "$output_file" | tail -1 | sed 's/.*Original size: *//' | sed 's/[^0-9].*//')
            fi
            if grep -q "Samples added:" "$output_file"; then
                augmentation_count=$(grep "Samples added:" "$output_file" | tail -1 | sed 's/.*Samples added: *//' | sed 's/[^0-9].*//')
            fi
        fi
        
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        # Add to CSV inventory
        echo "$dataset,$scenario,$method,$csv_path,$dataset_size,$original_size,$augmentation_count,$timestamp" >> "$CSV_INVENTORY"
        
        log_message "📊 CSV Generated: $csv_path (Size: $dataset_size, Added: $augmentation_count)"
    fi
}

experiment_count=0
total_experiments=$((${#DATASETS[@]} * ${#DISTRIBUTION_SCENARIOS[@]}))

# Main experiment loop
for dataset in "${DATASETS[@]}"; do
    log_message "Starting comprehensive analysis for $dataset..."
    
    # Create dataset-specific results directory
    dataset_results_dir="$RESULTS_DIR/${dataset}_faker"
    mkdir -p "$dataset_results_dir"
    
    for scenario_name in "${!DISTRIBUTION_SCENARIOS[@]}"; do
        scenario_value="${DISTRIBUTION_SCENARIOS[$scenario_name]}"
        
        if [[ "$scenario_name" == "original_ratios" ]]; then
            sensitive_ratio="original"
            label_ratio="original"
        else
            IFS=' ' read -r sensitive_ratio label_ratio <<< "$scenario_value"
        fi
        
        method="faker"
        log_message "[$((++experiment_count))/$total_experiments] Processing: $dataset - $scenario_name - $method"
        start_time=$(date +%s)
        
        output_file="$dataset_results_dir/${scenario_name}_faker.txt"
        
        # Build command array
        cmd=() 
        if [[ "$scenario_name" == "original_ratios" ]]; then
            # For original ratios, run standard ML experiment (no augmentation)
            cmd=(python main.py "yaml/${dataset}.yaml" \
                --fairness \
                --save_results \
                --results_folder "$dataset_results_dir" \
                --scenario_name "original_faker")
        else
            # For all other scenarios, run balanced augmentation experiment
            # The --balanced flag automatically triggers CSV saving in main.py
            cmd=(python main.py "yaml/${dataset}.yaml" \
                --balanced \
                --sensitive_ratio "$sensitive_ratio" \
                --label_ratio "$label_ratio" \
                --fairness \
                --save_results \
                --results_folder "$dataset_results_dir" \
                --scenario_name "${scenario_name}_faker" \
                --method "$method")
        fi

        # Execute experiment
        experiment_status=0
        log_message "🚀 Executing: ${cmd[*]}"
        "${cmd[@]}" > "$output_file" 2>&1
        experiment_status=$?
        
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # Process results
        status="ERROR"
        category="ERROR"
        
        if [[ $experiment_status -eq 0 && -f "$output_file" && -s "$output_file" ]]; then
            status="SUCCESS"
            
            # Extract metrics and CSV path
            metrics_result=$(extract_metrics "$output_file")
            IFS=',' read -r dp_improvement accuracy_cost final_dp_diff balance_achieved csv_path <<< "$metrics_result"
            
            # Categorize result
            category=$(categorize_result "$dp_improvement" "$accuracy_cost")
            
            # Track generated CSV
            track_generated_csv "$dataset" "$scenario_name" "$method" "$csv_path" "$output_file"
            
            log_message "✅ SUCCESS: $dataset-$scenario_name: DP=$dp_improvement, AC=$accuracy_cost, Cat=$category"
        else
            dp_improvement="N/A"
            accuracy_cost="N/A"
            final_dp_diff="N/A"
            balance_achieved="No"
            csv_path="N/A"
            log_message "❌ ERROR: $dataset-$scenario_name failed (exit code: $experiment_status)"
            
            # Log error details
            if [[ -f "$output_file" ]]; then
                echo "Error details:" >> "$LOG_FILE"
                tail -20 "$output_file" >> "$LOG_FILE"
            fi
        fi
        
        # Write to results matrix
        echo "$dataset,$scenario_name,$sensitive_ratio,$label_ratio,$method,$dp_improvement,$accuracy_cost,$category,$final_dp_diff,$balance_achieved,$status,$duration,$csv_path" >> "$RESULTS_MATRIX"
        
        sleep 1
    done
done

# Create comprehensive analysis report
ANALYSIS_REPORT="$RESULTS_DIR/comprehensive_analysis_report.txt"
cat > "$ANALYSIS_REPORT" << EOF
COMPREHENSIVE IMBALANCE IMPACT ANALYSIS REPORT
==============================================
Generated: $(date)
Total Experiments: $total_experiments

EXPERIMENT CONFIGURATION:
- Datasets: ${DATASETS[*]}
- Methods: ${METHODS[*]}
- Scenarios: $(echo "${!DISTRIBUTION_SCENARIOS[@]}" | tr ' ' '\n' | sort | tr '\n' ' ')

CSV GENERATION STATUS:
=====================
EOF

# Count CSV files generated
csv_count=$(grep -c "^[^,]*,[^,]*,[^,]*,[^N/A]" "$CSV_INVENTORY" 2>/dev/null || echo "0")
echo "Total CSV files generated: $csv_count" >> "$ANALYSIS_REPORT"

if [[ -f "$RESULTS_MATRIX" ]]; then
    # Results summary
    breakthrough_count=$(grep -c ",BREAKTHROUGH," "$RESULTS_MATRIX" 2>/dev/null || echo "0")
    moderate_count=$(grep -c ",MODERATE," "$RESULTS_MATRIX" 2>/dev/null || echo "0")
    high_cost_count=$(grep -c ",HIGH_ACCURACY_COST," "$RESULTS_MATRIX" 2>/dev/null || echo "0")
    unclear_count=$(grep -c ",UNCLEAR," "$RESULTS_MATRIX" 2>/dev/null || echo "0")
    failed_count=$(grep -c ",FAILED," "$RESULTS_MATRIX" 2>/dev/null || echo "0")
    error_count=$(grep -c ",ERROR," "$RESULTS_MATRIX" 2>/dev/null || echo "0")

    cat >> "$ANALYSIS_REPORT" << EOF

RESULTS CATEGORIZATION:
======================
- BREAKTHROUGH results: $breakthrough_count/$total_experiments (>3% DP improvement, <5% accuracy cost)
- MODERATE improvements: $moderate_count/$total_experiments (1-3% DP improvement)
- HIGH ACCURACY COST: $high_cost_count/$total_experiments (>3% DP improvement, ≥5% accuracy cost)
- UNCLEAR results: $unclear_count/$total_experiments (0-1% DP improvement)
- FAILED attempts: $failed_count/$total_experiments (≤0% DP improvement)
- ERROR cases: $error_count/$total_experiments

TOP PERFORMING SCENARIOS:
========================
EOF

    # Extract and display best results
    echo "Best DP improvements:" >> "$ANALYSIS_REPORT"
    {
        echo "Dataset,Scenario,DP_Improvement,Accuracy_Cost,Category"
        grep -v "ERROR" "$RESULTS_MATRIX" | grep -v "N/A" | grep -v "Dataset,Scenario" | sort -t',' -k6 -nr | head -10 | cut -d',' -f1,2,6,7,8
    } | column -t -s',' >> "$ANALYSIS_REPORT" 2>/dev/null || echo "No valid results found" >> "$ANALYSIS_REPORT"
    
fi

# Create CSV summary
if [[ -f "$CSV_INVENTORY" && $(wc -l < "$CSV_INVENTORY") -gt 1 ]]; then
    cat >> "$ANALYSIS_REPORT" << EOF

GENERATED DATASETS SUMMARY:
==========================
EOF
    echo "CSV Inventory ($(wc -l < "$CSV_INVENTORY") files):" >> "$ANALYSIS_REPORT"
    column -t -s',' "$CSV_INVENTORY" >> "$ANALYSIS_REPORT" 2>/dev/null || cat "$CSV_INVENTORY" >> "$ANALYSIS_REPORT"
fi

log_message "=== COMPREHENSIVE EXPERIMENT COMPLETED ==="
log_message "Results matrix: $RESULTS_MATRIX"
log_message "CSV inventory: $CSV_INVENTORY"
log_message "Analysis report: $ANALYSIS_REPORT"

echo ""
echo "🎯 COMPREHENSIVE IMBALANCE IMPACT ANALYSIS COMPLETED"
echo "📊 Results Matrix: $RESULTS_MATRIX"
echo "📁 CSV Inventory: $CSV_INVENTORY"
echo "📋 Analysis Report: $ANALYSIS_REPORT"
echo ""
echo "Key Features:"
echo "  ✅ Automatic CSV saving for all augmented datasets"
echo "  ✅ Comprehensive results tracking"
echo "  ✅ Dataset size and augmentation statistics"
echo "  ✅ Enhanced error handling and logging"
echo "  ✅ Detailed analysis report generation"
echo ""

# Final check and summary
total_csv_files=$(find "$RESULTS_DIR" -name "*.csv" -type f | wc -l)
log_message "Total CSV files in results directory: $total_csv_files"

if [[ $total_csv_files -gt 0 ]]; then
    echo "📈 Generated CSV files locations:"
    find "$RESULTS_DIR" -name "augmented_*.csv" -type f | head -10
    if [[ $(find "$RESULTS_DIR" -name "augmented_*.csv" -type f | wc -l) -gt 10 ]]; then
        echo "... and $(($(find "$RESULTS_DIR" -name "augmented_*.csv" -type f | wc -l) - 10)) more"
    fi
fi

echo ""
echo "🔍 To analyze the generated datasets:"
echo "  - Check the CSV inventory: $CSV_INVENTORY"
echo "  - Review the analysis report: $ANALYSIS_REPORT"
echo "  - Individual augmented datasets are in: $RESULTS_DIR/*/augmented_*.csv"
echo ""