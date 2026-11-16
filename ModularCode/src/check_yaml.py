import yaml
import os

base_path = os.path.dirname(os.path.abspath(__file__))  # directory containing check_yaml.py
yaml_dir = os.path.join(base_path, '..', 'yaml')  # go up one level, then into yaml

for dataset in ['africa', 'brazil', 'india']:
    with open(os.path.join(yaml_dir, f'{dataset}.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"{dataset}:")
    print(f"  sensitive_column: {config.get('sensitive_column', 'NOT SET')}")
    print(f"  label_column: {config.get('label_column', 'NOT SET')}")
    print()