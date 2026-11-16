#!/bin/bash

set -e

# Configuration
DATASETS="africa brazil india"
BASE_RESULTS_DIR="results"
ANALYSIS_SCRIPT="src/sensitive_attribute_analyzer.py"
EXE="python"

# YAML configurations
YAML_africa="yaml/africa.yaml"
YAML_brazil="yaml/brazil.yaml" 
YAML_india="yaml/india.yaml"

# Sweep values (same as used in generation)
SWEEP_VALUES="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95"

echo "🔄 Updating sensitive attribute analysis across all synthetic datasets..."
echo "Using updated analyzer: $ANALYSIS_SCRIPT"
echo ""

# Function to get YAML path for dataset
get_yaml_path() {
    case "$1" in
        brazil) echo "$YAML_brazil" ;;
        africa) echo "$YAML_africa" ;;
        india)  echo "$YAML_india" ;;
        *) echo ""; return 1 ;;
    esac
}

# Function to get original CSV path from YAML
get_original_csv() {
    local yaml_path="$1"
    awk -F': *' '/datapath:/ {gsub(/["'\'']/, "", $2); print $2}' "$yaml_path"
}

# Function to convert float to percentage (same as generation script)
float_to_percent() {
    local float_val="$1"
    python -c "print(int(round($float_val * 100)))"
}

# Function to update analysis for a specific experiment type
update_experiment_analysis() {
    local dataset="$1"
    local experiment_type="$2"  # "vary_sensitive" or "vary_label"
    local yaml_path="$3"
    local original_csv="$4"
    
    local experiment_dir="$BASE_RESULTS_DIR/$dataset/$experiment_type"
    local analysis_file="$experiment_dir/sensitive_attribute_analysis.csv"
    
    echo "[$dataset/$experiment_type] Updating analysis..."
    
    # Check if experiment directory exists
    if [ ! -d "$experiment_dir" ]; then
        echo "  ⚠️  Directory not found: $experiment_dir"
        return 1
    fi
    
    # Create new analysis file with header
    if [ "$experiment_type" = "vary_sensitive" ]; then
        echo "sensitive_ratio,original_privileged,original_unprivileged,original_priv_pct,synthetic_privileged,synthetic_unprivileged,synthetic_priv_pct,difference_pct" > "$analysis_file"
        fixed_label_ratio=0.50
        fixed_label_pct=$(float_to_percent $fixed_label_ratio)
    else
        echo "label_ratio,original_privileged,original_unprivileged,original_priv_pct,synthetic_privileged,synthetic_unprivileged,synthetic_priv_pct,difference_pct" > "$analysis_file"
        fixed_sens_ratio=0.50
        fixed_sens_pct=$(float_to_percent $fixed_sens_ratio)
    fi
    
    local processed_count=0
    local error_count=0
    
    # Process each sweep value
    for ratio in $SWEEP_VALUES; do
        ratio_pct=$(float_to_percent $ratio)
        
        if [ "$experiment_type" = "vary_sensitive" ]; then
            synthetic_file="$experiment_dir/${dataset}_Sens_${ratio_pct}_Label_${fixed_label_pct}.csv"
            ratio_value=$ratio
        else
            synthetic_file="$experiment_dir/${dataset}_Sens_${fixed_sens_pct}_Label_${ratio_pct}.csv"
            ratio_value=$ratio
        fi
        
        if [ -f "$synthetic_file" ]; then
            echo "    Processing $(basename "$synthetic_file")..."
            
            # Run the analyzer and capture output
            if $EXE $ANALYSIS_SCRIPT "$original_csv" "$synthetic_file" "$yaml_path" "$ratio_value" >> "$analysis_file" 2>/dev/null; then
                processed_count=$((processed_count + 1))
            else
                echo "      ❌ Analysis failed for $(basename "$synthetic_file")"
                error_count=$((error_count + 1))
                # Add error row to maintain CSV structure
                echo "$ratio_value,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$analysis_file"
            fi
        else
            echo "      ⚠️  Synthetic file not found: $(basename "$synthetic_file")"
            error_count=$((error_count + 1))
            # Add missing file row
            echo "$ratio_value,MISSING,MISSING,MISSING,MISSING,MISSING,MISSING,MISSING" >> "$analysis_file"
        fi
    done
    
    echo "    ✅ Completed: $processed_count successful, $error_count errors"
    echo "    📊 Analysis saved to: $analysis_file"
    
    return 0
}

# =============================================================================
# MAIN PROCESSING
# =============================================================================

total_datasets=0
successful_datasets=0

for dataset in $DATASETS; do
    echo ""
    echo "=== Processing Dataset: $dataset ==="
    
    # Get paths
    yaml_path=$(get_yaml_path "$dataset")
    if [ -z "$yaml_path" ] || [ ! -f "$yaml_path" ]; then
        echo "❌ YAML file not found: $yaml_path"
        continue
    fi
    
    original_csv=$(get_original_csv "$yaml_path")
    if [ ! -f "$original_csv" ]; then
        echo "❌ Original CSV not found: $original_csv"
        continue
    fi
    
    echo "Using YAML: $yaml_path"
    echo "Original data: $original_csv"
    
    total_datasets=$((total_datasets + 1))
    dataset_success=true
    
    # Update vary_sensitive analysis
    if ! update_experiment_analysis "$dataset" "vary_sensitive" "$yaml_path" "$original_csv"; then
        dataset_success=false
    fi
    
    # Update vary_label analysis  
    if ! update_experiment_analysis "$dataset" "vary_label" "$yaml_path" "$original_csv"; then
        dataset_success=false
    fi
    
    if [ "$dataset_success" = true ]; then
        successful_datasets=$((successful_datasets + 1))
        echo "✅ $dataset analysis update completed successfully"
    else
        echo "⚠️  $dataset analysis update completed with some issues"
    fi
done

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "🎉 ANALYSIS UPDATE COMPLETED 🎉"
echo "Processed $successful_datasets/$total_datasets datasets successfully"
echo ""

# Generate summary of updated files
echo "Summary of updated analysis files:"
for dataset in $DATASETS; do
    vary_sens_file="$BASE_RESULTS_DIR/$dataset/vary_sensitive/sensitive_attribute_analysis.csv"
    vary_label_file="$BASE_RESULTS_DIR/$dataset/vary_label/sensitive_attribute_analysis.csv"
    
    echo "[$dataset]:"
    
    if [ -f "$vary_sens_file" ]; then
        line_count=$(wc -l < "$vary_sens_file")
        echo "  ✅ vary_sensitive: $line_count lines (including header)"
        echo "     $vary_sens_file"
    else
        echo "  ❌ vary_sensitive: MISSING"
    fi
    
    if [ -f "$vary_label_file" ]; then
        line_count=$(wc -l < "$vary_label_file")
        echo "  ✅ vary_label: $line_count lines (including header)"
        echo "     $vary_label_file"
    else
        echo "  ❌ vary_label: MISSING"
    fi
done

echo ""
echo "Next steps:"
echo "1. Review the updated analysis CSV files for any ERROR or MISSING entries"
echo "2. Run visualization scripts to plot the fairness trends"
echo "3. Compare sensitive attribute distributions across different ratios"

# Optional: Show a sample of one analysis file
echo ""
echo "Sample from africa vary_sensitive analysis:"
if [ -f "$BASE_RESULTS_DIR/africa/vary_sensitive/sensitive_attribute_analysis.csv" ]; then
    echo "----------------------------------------"
    head -6 "$BASE_RESULTS_DIR/africa/vary_sensitive/sensitive_attribute_analysis.csv"
    echo "... (showing first 5 data rows)"
    echo "----------------------------------------"
fi