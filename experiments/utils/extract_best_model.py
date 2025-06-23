#!/usr/bin/env python3
"""
Extract best performing model from cross-validation output.
Usage: python extract_best_model.py <output_file>
"""

import sys
import re

def extract_best_model(output_file):
    """Extract the best model from cross-validation results."""
    
    models = {}
    
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Look for lines like: "logistic_regression: Mean accuracy = 0.8234, Std deviation = 0.0123"
        pattern = r'(\w+):\s+Mean accuracy = ([\d.]+)'
        matches = re.findall(pattern, content)
        
        for model_name, accuracy in matches:
            models[model_name] = float(accuracy)
        
        if models:
            # Return model with highest accuracy
            best_model = max(models, key=models.get)
            return best_model
        else:
            # Fallback if parsing fails
            return "logistic_regression"
    
    except Exception as e:
        print(f"Error parsing {output_file}: {e}", file=sys.stderr)
        return "logistic_regression"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_best_model.py <output_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    best_model = extract_best_model(output_file)
    print(best_model)