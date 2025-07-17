#!/bin/bash

# =================================================================
# UNIT TEST 2: PURE LLM SYNTHETIC DATA GENERATION
# Goal: Verify that the synchronous Pure LLM pipeline works correctly.
# =================================================================

set -e

echo "--- 🧪 STARTING PURE LLM GENERATION TEST ---"

# --- Test Parameters ---
CONFIG_FILE="yaml/brazil.yaml"
METHOD="llm_async" # This argument name is used by main.py
SENSITIVE_RATIO="0.7" # Use different ratios to test calculation
LABEL_RATIO="0.7"
SCENARIO_NAME="llm_unit_test"
RESULTS_DIR="./experiments/results/unit_tests"
API_KEY="sk-8d71fd82b1e34ce48e31718a1f3647bf" # <-- IMPORTANT: PASTE YOUR KEY HERE

# Validate API Key
if [ "$API_KEY" == "YOUR_API_KEY_HERE" ]; then
    echo "❌ ERROR: Please replace 'YOUR_API_KEY_HERE' with your actual API key in this script."
    exit 1
fi

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
# This will call the PureLLMSyntheticGenerator and make real API calls.
# It will be slower than the Faker test.
python main.py "$CONFIG_FILE" \
    --balanced \
    --fairness \
    --method "$METHOD" \
    --sensitive_ratio "$SENSITIVE_RATIO" \
    --label_ratio "$LABEL_RATIO" \
    --api_key "$API_KEY" \
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

echo "--- ✅ PURE LLM GENERATION TEST PASSED ---"
echo ""