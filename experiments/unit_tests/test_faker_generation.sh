#!/bin/bash

# =================================================================
# UNIT TEST 1: FAKER-BASED SYNTHETIC DATA GENERATION
# Goal: Verify that the synchronous Faker pipeline works correctly.
# =================================================================

set -e

echo "--- 🧪 STARTING FAKER GENERATION TEST ---"

# --- Test Parameters ---
CONFIG_FILE="yaml/brazil.yaml"
METHOD="faker"
SENSITIVE_RATIO="0.5"
LABEL_RATIO="0.5"
SCENARIO_NAME="faker_unit_test"
RESULTS_DIR="./experiments/results/unit_tests"

# Clean up previous test results
rm -rf "$RESULTS_DIR/$SCENARIO_NAME"
mkdir -p "$RESULTS_DIR"

echo "Running main.py with the following parameters:"
echo "  Config: $CONFIG_FILE"
echo "  Method: $METHOD"
echo "  Target Balance: $SENSITIVE_RATIO (Sensitive), $LABEL_RATIO (Label)"
echo "  Scenario: $SCENARIO_NAME"
echo "-------------------------------------------------"

# --- Execute the Test ---
# We will call main.py directly to generate a balanced dataset.
# The logic within main.py will calculate that we need to add many samples.
# We are just verifying that the call succeeds and produces output.
python main.py "$CONFIG_FILE" \
    --balanced \
    --fairness \
    --method "$METHOD" \
    --sensitive_ratio "$SENSITIVE_RATIO" \
    --label_ratio "$LABEL_RATIO" \
    --scenario_name "$SCENARIO_NAME" \
    --save_results \
    --results_folder "$RESULTS_DIR"

# --- Verification ---
echo "-------------------------------------------------"
echo "VERIFICATION:"

# Check if the results JSON file was created
JSON_FILE_PATH="$RESULTS_DIR/balanced_augmentation_brazil_${SCENARIO_NAME}.json"
if [ -f "$JSON_FILE_PATH" ]; then
    echo "✅ SUCCESS: Results JSON file created at $JSON_FILE_PATH"
else
    echo "❌ FAILURE: Results JSON file was not created."
    exit 1
fi

# Check if the JSON file contains success metrics
if grep -q "dp_improvement" "$JSON_FILE_PATH"; then
    echo "✅ SUCCESS: JSON file contains 'dp_improvement' metric."
else
    echo "❌ FAILURE: JSON file is missing key metrics."
    exit 1
fi

echo "--- ✅ FAKER GENERATION TEST PASSED ---"
echo ""