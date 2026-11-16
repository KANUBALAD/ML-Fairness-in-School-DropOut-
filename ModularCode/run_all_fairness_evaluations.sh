#!/bin/bash

set -e

# Dataset configurations
DATASETS="brazil africa india"
YAML_brazil="yaml/brazil.yaml"
YAML_africa="yaml/africa.yaml" 
YAML_india="yaml/india.yaml"

# Cross-validation settings
CV_FOLDS=5

# Function to get original data path from YAML
get_original_from_yaml() {
    local yaml_file="$1"
    if [ -f "$yaml_file" ]; then
        awk -F': *' '/datapath:/ {gsub(/["'\'']/, "", $2); print $2}' "$yaml_file"
    else
        echo ""
    fi
}

# Function to count synthetic CSV files (excluding analysis files)
count_synthetic_files() {
    local folder="$1"
    local dataset="$2"
    
    if [ ! -d "$folder" ]; then
        echo 0
        return
    fi
    
    find "$folder" -name "${dataset}_Sens_*_Label_*.csv" -type f | \
    grep -v -E "(analysis|fairness_bulk_results)" | \
    wc -l
}

# Function to validate folder structure
validate_experiment_folder() {
    local folder="$1"
    local dataset="$2"
    local experiment_type="$3"
    
    echo "    Validating $experiment_type folder..."
    
    if [ ! -d "$folder" ]; then
        echo "      ❌ Directory not found: $folder"
        return 1
    fi
    
    local synthetic_count=$(count_synthetic_files "$folder" "$dataset")
    echo "      📁 Found $synthetic_count synthetic CSV files"
    
    if [ "$synthetic_count" -eq 0 ]; then
        echo "      ⚠️  No synthetic files found matching pattern: ${dataset}_Sens_*_Label_*.csv"
        return 1
    fi
    
    echo "      📋 File breakdown:"
    local total_csv=$(find "$folder" -name "*.csv" -type f | wc -l)
    local analysis_files=$(find "$folder" -name "*analysis*.csv" -type f | wc -l)
    local bulk_results=$(find "$folder" -name "fairness_bulk_results.csv" -type f | wc -l)
    
    echo "        Total CSV files: $total_csv"
    echo "        Analysis files (will skip): $analysis_files"
    echo "        Bulk results files (will skip): $bulk_results"
    echo "        Synthetic files (will process): $synthetic_count"
    
    echo "      📄 Sample synthetic files to process:"
    find "$folder" -name "${dataset}_Sens_*_Label_*.csv" -type f | \
    grep -v -E "(analysis|fairness_bulk_results)" | \
    head -3 | while read file; do
        echo "        $(basename "$file")"
    done
    
    if [ "$synthetic_count" -gt 3 ]; then
        echo "        ... and $(($synthetic_count - 3)) more files"
    fi
    
    return 0
}

# Function to show experiment results
show_experiment_results() {
    local out_csv="$1"
    local dataset="$2"
    local experiment_type="$3"
    
    if [ -f "$out_csv" ]; then
        line_count=$(wc -l < "$out_csv")
        file_size=$(du -h "$out_csv" | cut -f1)
        original_entries=$(grep -c ",original," "$out_csv" 2>/dev/null || echo "0")
        synthetic_entries=$(grep -c ",synthetic," "$out_csv" 2>/dev/null || echo "0")
        
        echo "      📈 Generated: $line_count lines, $file_size"
        echo "      📊 Entries: $original_entries original, $synthetic_entries synthetic"
        
        echo "      📋 Sample original results:"
        if [ "$original_entries" -gt 0 ]; then
            grep ",original," "$out_csv" | head -3 | while IFS=, read -r csv model dp_mean dp_std eo_mean eo_std acc_mean acc_std n cv_folds dataset_name type experiment sens_ratio label_ratio; do
                echo "        $model: Acc=${acc_mean}±${acc_std}, DP=${dp_mean}±${dp_std}, EO=${eo_mean}±${eo_std}"
            done
        else
            echo "        No original entries found"
        fi
    else
        echo "      ❌ Output file not found: $out_csv"
    fi
}

EXE="python"
BULK_SCRIPT="src/bulk_fairness_evaluation.py"
BASE_RESULTS_DIR="results"

echo "🔬 === Running Bulk Fairness Evaluations with Cross-Validation ==="
echo "Datasets: $DATASETS"
echo "CV Folds: $CV_FOLDS"
echo "Script: $BULK_SCRIPT"
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

# Track overall progress
total_experiments=0
successful_experiments=0

# Process each dataset
for dataset in $DATASETS; do
    echo "🗂️  === Processing Dataset: $dataset ==="
    
    YAML=$(get_yaml_path "$dataset")
    
    if [ -z "$YAML" ]; then
        echo "❌ ERROR: YAML configuration missing for dataset $dataset"
        continue
    fi
    
    ORIGINAL=$(get_original_from_yaml "$YAML")
    
    echo "  📋 Configuration:"
    echo "    YAML: $YAML"
    echo "    Original: $ORIGINAL"
    echo "    CV Folds: $CV_FOLDS"
    
    # Check if YAML and original file exist
    if [ ! -f "$YAML" ]; then
        echo "  ❌ ERROR: YAML file not found: $YAML"
        continue
    fi
    
    if [ ! -f "$ORIGINAL" ]; then
        echo "  ❌ ERROR: Original file not found: $ORIGINAL"
        echo "    (extracted from $YAML)"
        continue
    fi
    
    echo "  ✅ Configuration validated"
    echo ""
    
    # Process vary_sensitive experiment
    echo "  📊 Processing vary_sensitive experiment..."
    VARY_SENS_FOLDER="$BASE_RESULTS_DIR/$dataset/vary_sensitive"
    
    total_experiments=$((total_experiments + 1))
    
    if validate_experiment_folder "$VARY_SENS_FOLDER" "$dataset" "vary_sensitive"; then
        OUT_CSV="$VARY_SENS_FOLDER/fairness_bulk_results.csv"
        
        echo "    🚀 Running cross-validated fairness evaluation..."
        echo "      Input folder: $VARY_SENS_FOLDER"
        echo "      Output file: $OUT_CSV"
        echo "      Cross-validation: $CV_FOLDS folds"
        
        if $EXE $BULK_SCRIPT \
            --config "$YAML" \
            --folder "$VARY_SENS_FOLDER" \
            --original "$ORIGINAL" \
            --out_csv "$OUT_CSV" \
            --cv_folds "$CV_FOLDS"; then
            
            echo "    ✅ vary_sensitive evaluation completed!"
            echo "      📊 Results saved to: $OUT_CSV"
            
            show_experiment_results "$OUT_CSV" "$dataset" "vary_sensitive"
            successful_experiments=$((successful_experiments + 1))
        else
            echo "    ❌ Failed to process vary_sensitive for $dataset"
        fi
    else
        echo "    ⚠️  Skipping vary_sensitive (validation failed)"
    fi
    
    echo ""
    
    # Process vary_label experiment  
    echo "  📊 Processing vary_label experiment..."
    VARY_LABEL_FOLDER="$BASE_RESULTS_DIR/$dataset/vary_label"
    
    total_experiments=$((total_experiments + 1))
    
    if validate_experiment_folder "$VARY_LABEL_FOLDER" "$dataset" "vary_label"; then
        OUT_CSV="$VARY_LABEL_FOLDER/fairness_bulk_results.csv"
        
        echo "    🚀 Running cross-validated fairness evaluation..."
        echo "      Input folder: $VARY_LABEL_FOLDER"
        echo "      Output file: $OUT_CSV"
        echo "      Cross-validation: $CV_FOLDS folds"
        
        if $EXE $BULK_SCRIPT \
            --config "$YAML" \
            --folder "$VARY_LABEL_FOLDER" \
            --original "$ORIGINAL" \
            --out_csv "$OUT_CSV" \
            --cv_folds "$CV_FOLDS"; then
            
            echo "    ✅ vary_label evaluation completed!"
            echo "      📊 Results saved to: $OUT_CSV"
            
            show_experiment_results "$OUT_CSV" "$dataset" "vary_label"
            successful_experiments=$((successful_experiments + 1))
        else
            echo "    ❌ Failed to process vary_label for $dataset"
        fi
    else
        echo "    ⚠️  Skipping vary_label (validation failed)"
    fi
    
    echo ""
done

# Final summary
echo "🎉 === ALL FAIRNESS EVALUATIONS COMPLETED ==="
echo "📊 Summary: $successful_experiments/$total_experiments experiments processed successfully"
echo ""

if [ $successful_experiments -gt 0 ]; then
    echo "📂 Generated cross-validated fairness evaluation results:"
    
    # Summary table header
    printf "%-10s %-15s %-10s %-15s %-15s\n" "Dataset" "Experiment" "Lines" "Original" "Synthetic"
    printf "%-10s %-15s %-10s %-15s %-15s\n" "--------" "---------------" "------" "---------" "---------"
    
    for dataset in $DATASETS; do
        # Check vary_sensitive results
        vary_sens_file="$BASE_RESULTS_DIR/$dataset/vary_sensitive/fairness_bulk_results.csv"
        if [ -f "$vary_sens_file" ]; then
            line_count=$(wc -l < "$vary_sens_file")
            original_count=$(grep -c ",original," "$vary_sens_file" 2>/dev/null || echo "0")
            synthetic_count=$(grep -c ",synthetic," "$vary_sens_file" 2>/dev/null || echo "0")
            printf "%-10s %-15s %-10s %-15s %-15s\n" "$dataset" "vary_sensitive" "$line_count" "$original_count" "$synthetic_count"
        else
            printf "%-10s %-15s %-10s %-15s %-15s\n" "$dataset" "vary_sensitive" "MISSING" "-" "-"
        fi
        
        # Check vary_label results
        vary_label_file="$BASE_RESULTS_DIR/$dataset/vary_label/fairness_bulk_results.csv"
        if [ -f "$vary_label_file" ]; then
            line_count=$(wc -l < "$vary_label_file")
            original_count=$(grep -c ",original," "$vary_label_file" 2>/dev/null || echo "0")
            synthetic_count=$(grep -c ",synthetic," "$vary_label_file" 2>/dev/null || echo "0")
            printf "%-10s %-15s %-10s %-15s %-15s\n" "$dataset" "vary_label" "$line_count" "$original_count" "$synthetic_count"
        else
            printf "%-10s %-15s %-10s %-15s %-15s\n" "$dataset" "vary_label" "MISSING" "-" "-"
        fi
    done
    
    echo ""
    echo "📂 File locations:"
    for dataset in $DATASETS; do
        echo "  $dataset:"
        vary_sens_file="$BASE_RESULTS_DIR/$dataset/vary_sensitive/fairness_bulk_results.csv"
        vary_label_file="$BASE_RESULTS_DIR/$dataset/vary_label/fairness_bulk_results.csv"
        
        if [ -f "$vary_sens_file" ]; then
            echo "    ✅ vary_sensitive: $vary_sens_file"
        else
            echo "    ❌ vary_sensitive: MISSING"
        fi
        
        if [ -f "$vary_label_file" ]; then
            echo "    ✅ vary_label: $vary_label_file"
        else
            echo "    ❌ vary_label: MISSING"
        fi
    done
    
    echo ""
    echo "🎯 Next steps:"
    echo "1. Review fairness_bulk_results.csv files across all datasets"
    echo "2. Compare cross-validated DP and EO metrics with confidence intervals"
    echo "3. Analyze accuracy vs fairness trade-offs statistically"
    echo "4. Identify best synthetic configurations per dataset"
    echo "5. Generate comparative fairness visualizations"
    
else
    echo "❌ No experiments completed successfully. Check the errors above."
fi

# Show current status
echo ""
echo "📊 Current Results Status:"
for dataset in $DATASETS; do
    sens_file="$BASE_RESULTS_DIR/$dataset/vary_sensitive/fairness_bulk_results.csv"
    label_file="$BASE_RESULTS_DIR/$dataset/vary_label/fairness_bulk_results.csv"
    
    sens_status="❌"
    label_status="❌"
    
    if [ -f "$sens_file" ]; then
        sens_status="✅"
    fi
    
    if [ -f "$label_file" ]; then
        label_status="✅"
    fi
    
    echo "  $dataset: vary_sensitive $sens_status, vary_label $label_status"
done