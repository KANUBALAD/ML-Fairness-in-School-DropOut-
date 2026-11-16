#!/bin/bash

set -e

# Dataset configurations (now includes all three datasets)
DATASETS="africa brazil india"
YAML_africa="yaml/africa.yaml"
YAML_brazil="yaml/brazil.yaml"
YAML_india="yaml/india.yaml"

# Sweep ranges
SWEEP_VALUES="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95"

# Script settings
EXE="python"
CTGAN_SCRIPT="src/CTGAN_synthetic_generator.py"
ANALYSIS_SCRIPT="src/sensitive_attribute_analyzer.py"
BASE_RESULTS_DIR="results"
EPOCHS=50
BATCH_SIZE=100

echo "Starting comprehensive fairness experiments..."
echo "Datasets: $DATASETS"
echo "Using updated YAML configurations with explicit sensitive/label columns"
echo ""

# Function to convert float to percentage (robust alternative to bc)
float_to_percent() {
    local float_val="$1"
    # Use python for reliable float arithmetic
    python -c "print(int(round($float_val * 100)))"
}

# Function to get YAML path for dataset
get_yaml_path() {
    case "$1" in
        brazil) echo "$YAML_brazil" ;;
        africa) echo "$YAML_africa" ;;
        india)  echo "$YAML_india" ;;
        *) echo ""; return 1 ;;
    esac
}

# Function to get total rows from CSV (excluding header)
get_total_rows() {
    local csv_file="$1"
    echo $(($(wc -l < "$csv_file") - 1))
}

# Function to validate dataset setup
validate_dataset() {
    local dataset="$1"
    local yaml_path="$2"
    
    echo "Validating $dataset setup..."
    
    if [ ! -f "$yaml_path" ]; then
        echo "❌ ERROR: YAML file not found: $yaml_path"
        return 1
    fi
    
    # Extract CSV path from YAML and validate
    local csv_file=$(awk -F': *' '/datapath:/ {gsub(/["'\'']/, "", $2); print $2}' "$yaml_path")
    if [ ! -f "$csv_file" ]; then
        echo "❌ ERROR: Data file not found: $csv_file"
        return 1
    fi
    
    # Check if YAML has sensitive_column and label_column
    local sens_col=$(awk -F': *' '/sensitive_column:/ {gsub(/["'\'']/, "", $2); print $2}' "$yaml_path")
    local label_col=$(awk -F': *' '/label_column:/ {gsub(/["'\'']/, "", $2); print $2}' "$yaml_path")
    
    if [ -z "$sens_col" ] || [ -z "$label_col" ]; then
        echo "⚠️  WARNING: $dataset YAML missing sensitive_column or label_column"
        echo "   Will use auto-detection (may not be optimal)"
    else
        echo "✅ $dataset: sensitive_column='$sens_col', label_column='$label_col'"
    fi
    
    local total_rows=$(get_total_rows "$csv_file")
    echo "✅ $dataset: $total_rows rows in $csv_file"
    
    return 0
}

# =============================================================================
# VALIDATION PHASE
# =============================================================================
echo "=== VALIDATION PHASE ==="
all_valid=true

for dataset in $DATASETS; do
    YAML=$(get_yaml_path "$dataset")
    if [ -z "$YAML" ]; then
        echo "❌ ERROR: Unknown dataset $dataset"
        all_valid=false
        continue
    fi
    
    if ! validate_dataset "$dataset" "$YAML"; then
        all_valid=false
    fi
done

if [ "$all_valid" != true ]; then
    echo ""
    echo "❌ VALIDATION FAILED - Please fix the issues above before running experiments"
    exit 1
fi

echo ""
echo "✅ ALL DATASETS VALIDATED - Proceeding with experiments"
echo ""

# =============================================================================
# EXPERIMENT 1: Vary Sensitive Ratio (Fixed Label Ratio = 0.50)
# =============================================================================
echo "=== EXPERIMENT 1: Varying Sensitive Ratio (Label fixed at 0.50) ==="

for dataset in $DATASETS; do
    YAML=$(get_yaml_path "$dataset")
    
    # Extract CSV path from YAML
    CSV_FILE=$(awk -F': *' '/datapath:/ {gsub(/["'\'']/, "", $2); print $2}' "$YAML")
    TOTAL_ROWS=$(get_total_rows "$CSV_FILE")
    
    # Setup output directory
    OUTDIR="$BASE_RESULTS_DIR/$dataset/vary_sensitive"
    mkdir -p "$OUTDIR"
    
    # Fixed label ratio for this experiment
    FIXED_LABEL_RATIO=0.50
    FIXED_LABEL_PCT=$(float_to_percent $FIXED_LABEL_RATIO)
    
    echo ""
    echo "[$dataset] Processing vary_sensitive experiment..."
    echo "  YAML: $YAML"
    echo "  CSV: $CSV_FILE ($TOTAL_ROWS rows)"
    echo "  Output: $OUTDIR"
    echo "  Fixed label ratio: $FIXED_LABEL_RATIO ($FIXED_LABEL_PCT%)"
    
    # Create analysis summary file
    ANALYSIS_FILE="$OUTDIR/sensitive_attribute_analysis.csv"
    echo "sensitive_ratio,original_privileged,original_unprivileged,original_priv_pct,synthetic_privileged,synthetic_unprivileged,synthetic_priv_pct,difference_pct" > "$ANALYSIS_FILE"
    
    for SENS_RATIO in $SWEEP_VALUES; do
        SENS_PCT=$(float_to_percent $SENS_RATIO)
        EXPECTED_FILE="$OUTDIR/${dataset}_Sens_${SENS_PCT}_Label_${FIXED_LABEL_PCT}.csv"
        
        echo "  Processing sensitive_ratio = $SENS_RATIO (${SENS_PCT}%)"
        echo "    Expected output: $(basename "$EXPECTED_FILE")"
        
        $EXE $CTGAN_SCRIPT $YAML \
            --sensitive_ratio $SENS_RATIO \
            --label_ratio $FIXED_LABEL_RATIO \
            --total_rows $TOTAL_ROWS \
            --outdir "$OUTDIR" \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE
            
        if [ $? -ne 0 ]; then
            echo "  ⚠️  ERROR: Failed for $dataset with sensitive_ratio=$SENS_RATIO"
            continue
        fi
        
        # Verify file was created with expected name
        if [ -f "$EXPECTED_FILE" ]; then
            echo "  ✅ Generated: $(basename "$EXPECTED_FILE")"
            
            # Run analysis
            if [ -f "$ANALYSIS_SCRIPT" ]; then
                echo "  📊 Running analysis..."
                $EXE $ANALYSIS_SCRIPT "$CSV_FILE" "$EXPECTED_FILE" "$YAML" "$SENS_RATIO" >> "$ANALYSIS_FILE"
            fi
        else
            echo "  ❌ Expected file not found: $EXPECTED_FILE"
            # List what files were actually created
            echo "     Files in output directory:"
            ls -la "$OUTDIR"/${dataset}_Sens_*_Label_*.csv 2>/dev/null || echo "     No matching files found"
        fi
    done
    
    echo "  ✅ Completed vary_sensitive for $dataset"
    echo "  📊 Analysis saved to: $ANALYSIS_FILE"
    echo "  📁 Generated files: $(ls "$OUTDIR"/${dataset}_Sens_*_Label_${FIXED_LABEL_PCT}.csv 2>/dev/null | wc -l) CSV files"
done

# =============================================================================
# EXPERIMENT 2: Vary Label Ratio (Fixed Sensitive Ratio = 0.50)
# =============================================================================
echo ""
echo "=== EXPERIMENT 2: Varying Label Ratio (Sensitive fixed at 0.50) ==="

for dataset in $DATASETS; do
    YAML=$(get_yaml_path "$dataset")
    
    # Extract CSV path from YAML
    CSV_FILE=$(awk -F': *' '/datapath:/ {gsub(/["'\'']/, "", $2); print $2}' "$YAML")
    TOTAL_ROWS=$(get_total_rows "$CSV_FILE")
    
    # Setup output directory
    OUTDIR="$BASE_RESULTS_DIR/$dataset/vary_label"
    mkdir -p "$OUTDIR"
    
    # Fixed sensitive ratio for this experiment  
    FIXED_SENS_RATIO=0.50
    FIXED_SENS_PCT=$(float_to_percent $FIXED_SENS_RATIO)
    
    echo ""
    echo "[$dataset] Processing vary_label experiment..."
    echo "  YAML: $YAML"
    echo "  CSV: $CSV_FILE ($TOTAL_ROWS rows)"
    echo "  Output: $OUTDIR"
    echo "  Fixed sensitive ratio: $FIXED_SENS_RATIO ($FIXED_SENS_PCT%)"
    
    # Create analysis summary file
    ANALYSIS_FILE="$OUTDIR/sensitive_attribute_analysis.csv"
    echo "label_ratio,original_privileged,original_unprivileged,original_priv_pct,synthetic_privileged,synthetic_unprivileged,synthetic_priv_pct,difference_pct" > "$ANALYSIS_FILE"
    
    for LABEL_RATIO in $SWEEP_VALUES; do
        LABEL_PCT=$(float_to_percent $LABEL_RATIO)
        EXPECTED_FILE="$OUTDIR/${dataset}_Sens_${FIXED_SENS_PCT}_Label_${LABEL_PCT}.csv"
        
        echo "  Processing label_ratio = $LABEL_RATIO (${LABEL_PCT}%)"
        echo "    Expected output: $(basename "$EXPECTED_FILE")"
        
        $EXE $CTGAN_SCRIPT $YAML \
            --sensitive_ratio $FIXED_SENS_RATIO \
            --label_ratio $LABEL_RATIO \
            --total_rows $TOTAL_ROWS \
            --outdir "$OUTDIR" \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE
            
        if [ $? -ne 0 ]; then
            echo "  ⚠️  ERROR: Failed for $dataset with label_ratio=$LABEL_RATIO"
            continue
        fi
        
        # Verify file was created with expected name
        if [ -f "$EXPECTED_FILE" ]; then
            echo "  ✅ Generated: $(basename "$EXPECTED_FILE")"
            
            # Run analysis
            if [ -f "$ANALYSIS_SCRIPT" ]; then
                echo "  📊 Running analysis..."
                $EXE $ANALYSIS_SCRIPT "$CSV_FILE" "$EXPECTED_FILE" "$YAML" "$LABEL_RATIO" >> "$ANALYSIS_FILE"
            fi
        else
            echo "  ❌ Expected file not found: $EXPECTED_FILE"
            # List what files were actually created
            echo "     Files in output directory:"
            ls -la "$OUTDIR"/${dataset}_Sens_*_Label_*.csv 2>/dev/null || echo "     No matching files found"
        fi
    done
    
    echo "  ✅ Completed vary_label for $dataset"
    echo "  📊 Analysis saved to: $ANALYSIS_FILE"
    echo "  📁 Generated files: $(ls "$OUTDIR"/${dataset}_Sens_${FIXED_SENS_PCT}_Label_*.csv 2>/dev/null | wc -l) CSV files"
done

# =============================================================================
# FINAL SUMMARY
# =============================================================================
echo ""
echo "🎉 ALL EXPERIMENTS COMPLETED 🎉"
echo "Results saved in: $BASE_RESULTS_DIR/"
echo ""

# Generate detailed summary
for dataset in $DATASETS; do
    echo "[$dataset] Summary:"
    vary_sens_count=$(find "$BASE_RESULTS_DIR/$dataset/vary_sensitive" -name "${dataset}_Sens_*_Label_50.csv" 2>/dev/null | wc -l)
    vary_label_count=$(find "$BASE_RESULTS_DIR/$dataset/vary_label" -name "${dataset}_Sens_50_Label_*.csv" 2>/dev/null | wc -l)
    
    echo "  📁 Vary sensitive: $vary_sens_count files in $BASE_RESULTS_DIR/$dataset/vary_sensitive/"
    echo "  📁 Vary label:     $vary_label_count files in $BASE_RESULTS_DIR/$dataset/vary_label/"
    
    # Show sample file names
    if [ $vary_sens_count -gt 0 ]; then
        echo "     Sample files:"
        find "$BASE_RESULTS_DIR/$dataset/vary_sensitive" -name "${dataset}_Sens_*_Label_50.csv" 2>/dev/null | head -3 | while read file; do
            echo "       $(basename "$file")"
        done
    fi
done

echo ""
echo "Expected folder structure verification:"
echo "tree $BASE_RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Check the generated files match your expected naming pattern"
echo "2. Review analysis CSV files for attribute distribution accuracy"
echo "3. Run visualization scripts to see fairness trends"
echo "4. Use generated synthetic data for ML model training"