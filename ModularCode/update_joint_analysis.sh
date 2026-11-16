#!/bin/bash

set -e

# Configuration
DATASETS="africa brazil india"
BASE_RESULTS_DIR="results"
JOINT_ANALYSIS_SCRIPT="analyze_fairness_joint.py"
EXE="python"

# YAML configurations
YAML_africa="yaml/africa.yaml"
YAML_brazil="yaml/brazil.yaml"
YAML_india="yaml/india.yaml"

echo "🔄 Updating joint distribution analysis across all synthetic datasets..."
echo "Using updated joint analyzer: $JOINT_ANALYSIS_SCRIPT"
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

# Function to update joint analysis for a specific experiment type
update_joint_analysis() {
    local dataset="$1"
    local experiment_type="$2"  # "vary_sensitive" or "vary_label"
    local yaml_path="$3"
    local original_csv="$4"
    
    local experiment_dir="$BASE_RESULTS_DIR/$dataset/$experiment_type"
    local output_file="$experiment_dir/joint_distribution_analysis.csv"
    
    echo "[$dataset/$experiment_type] Updating joint distribution analysis..."
    
    # Check if experiment directory exists
    if [ ! -d "$experiment_dir" ]; then
        echo "  ⚠️  Directory not found: $experiment_dir"
        return 1
    fi
    
    # Count synthetic CSV files (excluding analysis files)
    synthetic_count=$(find "$experiment_dir" -name "${dataset}_Sens_*_Label_*.csv" -type f | wc -l)
    echo "  Found $synthetic_count synthetic files to analyze"
    
    if [ $synthetic_count -eq 0 ]; then
        echo "  ⚠️  No synthetic files found in $experiment_dir"
        return 1
    fi
    
    # Run the joint analysis
    echo "  Running joint distribution analysis..."
    echo "    Input directory: $experiment_dir"
    echo "    Original data: $original_csv"
    echo "    YAML config: $yaml_path"
    echo "    Output: $output_file"
    
    if $EXE $JOINT_ANALYSIS_SCRIPT "$experiment_dir" "$original_csv" "$yaml_path"; then
        echo "  ✅ Joint analysis completed successfully"
        
        # Verify output file was created and show summary
        if [ -f "$output_file" ]; then
            line_count=$(wc -l < "$output_file")
            echo "  📊 Generated analysis with $line_count lines (including header)"
            echo "  📁 Output saved to: $output_file"
        else
            echo "  ⚠️  Expected output file not found: $output_file"
            return 1
        fi
    else
        echo "  ❌ Joint analysis failed for $dataset/$experiment_type"
        return 1
    fi
    
    return 0
}

# =============================================================================
# MAIN PROCESSING
# =============================================================================

total_experiments=0
successful_experiments=0

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
    
    # Process both experiment types
    for experiment_type in "vary_sensitive" "vary_label"; do
        total_experiments=$((total_experiments + 1))
        
        if update_joint_analysis "$dataset" "$experiment_type" "$yaml_path" "$original_csv"; then
            successful_experiments=$((successful_experiments + 1))
        fi
    done
done

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "🎉 JOINT DISTRIBUTION ANALYSIS UPDATE COMPLETED 🎉"
echo "Successfully processed $successful_experiments/$total_experiments experiment directories"
echo ""

# Generate summary of updated files
echo "Summary of updated joint distribution analysis files:"
for dataset in $DATASETS; do
    echo "[$dataset]:"
    
    for experiment_type in "vary_sensitive" "vary_label"; do
        analysis_file="$BASE_RESULTS_DIR/$dataset/$experiment_type/joint_distribution_analysis.csv"
        
        if [ -f "$analysis_file" ]; then
            line_count=$(wc -l < "$analysis_file")
            file_size=$(du -h "$analysis_file" | cut -f1)
            echo "  ✅ $experiment_type: $line_count lines, $file_size"
            echo "     $analysis_file"
        else
            echo "  ❌ $experiment_type: MISSING"
        fi
    done
done

echo ""
echo "Next steps:"
echo "1. Review the joint_distribution_analysis.csv files for any ERROR entries"
echo "2. Check that all four groups (sensitive×label combinations) are represented"
echo "3. Use these files for fairness visualization and reporting"
echo "4. Compare joint distributions across different ratio settings"

# Optional: Show a sample from one analysis file
sample_file="$BASE_RESULTS_DIR/africa/vary_sensitive/joint_distribution_analysis.csv"
if [ -f "$sample_file" ]; then
    echo ""
    echo "Sample from africa vary_sensitive joint analysis:"
    echo "------------------------------------------------"
    head -6 "$sample_file" | while IFS= read -r line; do
        echo "  $line"
    done
    echo "  ... (showing first 5 data rows)"
    echo "------------------------------------------------"
fi