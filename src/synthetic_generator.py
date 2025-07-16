# ModularCode/src/synthetic_generator.py (v2.1 - Fixed KeyError)

import re
import pandas as pd
import numpy as np
from faker import Faker
import json
import random
from typing import Dict, List, Tuple, Optional, Union, Any
import time
from .dynamic_schema_extractor import DynamicSchemaExtractor

class SyntheticDataGenerator:
    """
    Faker-based synthetic data generator with dynamic schema extraction.
    Automatically learns data distributions and generates realistic synthetic data.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize the synthetic data generator with dynamic schema capabilities."""
        self.faker = Faker(['en_US', 'pt_BR', 'hi_IN', 'en_GB', 'fr_FR'])  # Multi-locale support
        
        # Initialize dynamic schema extractor
        self.schema_extractor = DynamicSchemaExtractor()
        self._dataset_schemas = {}  # Cache for extracted schemas
        self._column_mappings = {}  # Cache for column mappings

    def _get_or_extract_schema(self, dataset_name: str, reference_data: pd.DataFrame) -> Dict:
        """Get cached schema or extract new one."""
        
        if dataset_name not in self._dataset_schemas:
            print(f"  Extracting dynamic schema for Faker generation: {dataset_name}")
            schema = self.schema_extractor.extract_comprehensive_schema(reference_data)
            self._dataset_schemas[dataset_name] = schema
            
            # Also extract column mappings
            self._column_mappings[dataset_name] = self._identify_key_columns(schema, reference_data)
            print(f"  ✓ Schema extracted and cached")
        
        return self._dataset_schemas[dataset_name]

    def _identify_key_columns(self, schema: Dict, reference_data: pd.DataFrame) -> Dict[str, str]:
        """Identify sensitive and label columns from the schema."""
        
        relationships = schema.get('relationships', {})
        sensitive_candidates = relationships.get('sensitive_column_candidates', [])
        label_candidates = relationships.get('label_column_candidates', [])
        
        # Use same logic as LLM generator for consistency
        sensitive_col = None
        label_col = None
        
        # Pick sensitive column
        if sensitive_candidates:
            for candidate in sensitive_candidates:
                if 'gender' in candidate.lower():
                    sensitive_col = candidate
                    break
            if not sensitive_col:
                sensitive_col = sensitive_candidates[0]
        
        # Pick label column  
        if label_candidates:
            priority_keywords = ['target', 'dropout', 'outcome', 'label']
            for keyword in priority_keywords:
                for candidate in label_candidates:
                    if keyword in candidate.lower():
                        label_col = candidate
                        break
                if label_col:
                    break
            if not label_col:
                label_col = label_candidates[0]

        # Fallback: manual detection if auto-detection fails
        if not sensitive_col:
            potential_sensitive = [col for col in reference_data.columns 
                                 if 'gender' in col.lower() or 'sex' in col.lower()]
            if potential_sensitive:
                sensitive_col = potential_sensitive[0]

        if not label_col:
            potential_labels = [col for col in reference_data.columns 
                              if any(keyword in col.lower() for keyword in ['target', 'dropout', 'outcome', 'label', 'class'])]
            if potential_labels:
                label_col = potential_labels[0]
        
        if not sensitive_col or not label_col:
            raise ValueError(f"Could not automatically identify key columns. Available columns: {list(reference_data.columns)}")
        
        print(f"  Identified key columns: sensitive='{sensitive_col}', label='{label_col}'")
        return {'sensitive_col': sensitive_col, 'label_col': label_col}

    def _get_column_values_safely(self, schema: Dict, reference_data: pd.DataFrame, col_name: str) -> List:
        """Safely get unique values for a column, handling both schema and direct data extraction."""
        
        # Try to get from schema first
        col_schema = schema['columns'].get(col_name, {})
        constraints = col_schema.get('constraints', {})
        
        # Check if values are available in schema
        if 'allowed_values' in constraints:
            values = constraints['allowed_values']
            if values:
                return values
        
        # Fallback: extract directly from data
        if col_name in reference_data.columns:
            unique_values = reference_data[col_name].dropna().unique().tolist()
            print(f"  Retrieved {len(unique_values)} unique values for '{col_name}' directly from data: {unique_values}")
            return unique_values
        
        raise ValueError(f"Could not extract values for column '{col_name}'")

    def _determine_category_values(self, schema: Dict, columns: Dict, category: str, reference_data: pd.DataFrame) -> Tuple:
        """Determine the specific values for a given category based on schema, with fallback to data extraction."""
        
        sensitive_col = columns['sensitive_col']
        label_col = columns['label_col']
        
        # Safely get values for both columns
        try:
            sensitive_values = self._get_column_values_safely(schema, reference_data, sensitive_col)
            label_values = self._get_column_values_safely(schema, reference_data, label_col)
        except Exception as e:
            print(f"  Error extracting column values: {e}")
            raise
        
        # Use same heuristics as LLM generator
        privileged_val, unprivileged_val = self._determine_privileged_values(sensitive_values, sensitive_col)
        positive_val, negative_val = self._determine_positive_values(label_values, label_col)
        
        print(f"  Value mapping: privileged={privileged_val}, unprivileged={unprivileged_val}, positive={positive_val}, negative={negative_val}")
        
        # Map category to values
        category_mapping = {
            'priv_pos': (privileged_val, positive_val),
            'priv_neg': (privileged_val, negative_val), 
            'unpriv_pos': (unprivileged_val, positive_val),
            'unpriv_neg': (unprivileged_val, negative_val)
        }
        
        return category_mapping.get(category, (sensitive_values[0], label_values[0]))

    def _determine_privileged_values(self, values: List, column_name: str) -> Tuple:
        """Same logic as LLM generator for consistency."""
        if len(values) < 2:
            return values[0] if values else 'Unknown', values[1] if len(values) > 1 else 'Unknown'
        
        val1, val2 = values[0], values[1]
        col_lower = column_name.lower()
        
        if 'gender' in col_lower or 'sex' in col_lower:
            val1_str, val2_str = str(val1).lower(), str(val2).lower()
            # Male (or 1) is typically considered privileged
            if val1_str in ['male', 'm', '1', 'man'] or val2_str in ['female', 'f', '0', 'woman']:
                return (val1, val2) if val1_str in ['male', 'm', '1', 'man'] else (val2, val1)
        
        # Default: use the first value as privileged (or sort if possible)
        try:
            sorted_vals = sorted(values[:2])
            # For numeric values, higher might be privileged
            if all(isinstance(v, (int, float)) for v in values[:2]):
                return max(val1, val2), min(val1, val2)
            else:
                return sorted_vals[1], sorted_vals[0]  # Reverse alphabetical
        except:
            return val1, val2

    def _determine_positive_values(self, values: List, column_name: str) -> Tuple:
        """Same logic as LLM generator for consistency.""" 
        if len(values) < 2:
            return values[0] if values else 'Unknown', values[1] if len(values) > 1 else 'Unknown'
        
        val1, val2 = values[0], values[1]
        col_lower = column_name.lower()
        
        # Dropout is typically the positive outcome (bad outcome we want to predict)
        if 'dropout' in col_lower:
            val1_str, val2_str = str(val1).lower(), str(val2).lower()
            if 'dropout' in val1_str and 'not' not in val1_str:
                return val1, val2
            elif 'dropout' in val2_str and 'not' not in val2_str:
                return val2, val1
            elif 'graduate' in val1_str:
                return val2, val1  # Graduate is negative, so dropout is positive
            elif 'graduate' in val2_str:
                return val1, val2
        
        if 'target' in col_lower:
            val1_str, val2_str = str(val1).lower(), str(val2).lower()
            if 'dropout' in val1_str:
                return val1, val2
            elif 'dropout' in val2_str:
                return val2, val1
        
        # Default: first alphabetically is positive
        try:
            sorted_vals = sorted(values[:2])
            return sorted_vals[0], sorted_vals[1]
        except:
            return val1, val2

    def generate_category_specific_samples(self, dataset_name: str, n_samples: int, 
                                         category: str, reference_data: pd.DataFrame) -> List[Dict]:
        """Generate samples using dynamic schema extraction."""
        
        print(f"  Generating {n_samples} samples for {category}")
        
        # Extract or get cached schema
        schema = self._get_or_extract_schema(dataset_name, reference_data)
        columns = self._column_mappings[dataset_name]
        
        # Determine target values for this category (now with reference_data fallback)
        target_sensitive, target_label = self._determine_category_values(schema, columns, category, reference_data)
        
        print(f"    Target: {columns['sensitive_col']}={target_sensitive}, {columns['label_col']}={target_label}")
        
        # Generate samples using the schema
        samples = []
        for i in range(n_samples):
            sample = self._generate_schema_based_sample(
                schema, columns, target_sensitive, target_label, reference_data
            )
            samples.append(sample)
        
        print(f"    ✓ Generated {len(samples)} samples")
        return samples

    def _generate_schema_based_sample(self, schema: Dict, columns: Dict, 
                                    target_sensitive, target_label, reference_data: pd.DataFrame) -> Dict:
        """Generate a single sample based on the extracted schema."""
        
        sample = {}
        
        for col_name in reference_data.columns:  # Use actual columns from data
            col_schema = schema['columns'].get(col_name, {})
            type_category = col_schema.get('type_category', 'unknown')
            
            # Handle target columns first
            if col_name == columns['sensitive_col']:
                sample[col_name] = target_sensitive
                continue
            elif col_name == columns['label_col']:
                sample[col_name] = target_label
                continue
            
            # Generate based on column type
            if type_category == 'categorical':
                sample[col_name] = self._generate_categorical_value(col_schema, reference_data, col_name)
            elif type_category == 'numeric':
                sample[col_name] = self._generate_numeric_value(col_schema, reference_data, col_name)
            elif type_category == 'datetime':
                sample[col_name] = self._generate_datetime_value(col_schema, reference_data, col_name)
            elif type_category == 'text':
                sample[col_name] = self._generate_text_value(col_schema, reference_data, col_name)
            else:
                # Fallback for unknown types
                sample[col_name] = self._generate_fallback_value(col_name, reference_data)
        
        return sample

    def _generate_categorical_value(self, col_schema: Dict, reference_data: pd.DataFrame, col_name: str) -> Any:
        """Generate a categorical value based on schema constraints."""
        
        constraints = col_schema.get('constraints', {})
        allowed_values = constraints.get('allowed_values', [])
        
        # Fallback to data if no allowed values in schema
        if not allowed_values and col_name in reference_data.columns:
            allowed_values = reference_data[col_name].dropna().unique().tolist()
        
        if not allowed_values:
            return 'Unknown'
        
        # Use weighted selection based on original distribution
        value_counts = constraints.get('value_counts', {})
        if value_counts and all(val in allowed_values for val in value_counts.keys()):
            # Weighted random selection
            values = list(value_counts.keys())
            weights = list(value_counts.values())
            return random.choices(values, weights=weights)[0]
        else:
            return random.choice(allowed_values)

    def _generate_numeric_value(self, col_schema: Dict, reference_data: pd.DataFrame, col_name: str) -> Union[int, float]:
        """Generate a numeric value based on schema constraints."""
        
        constraints = col_schema.get('constraints', {})
        distribution_info = col_schema.get('distribution_info', {})
        
        # Check if specific values are allowed
        unique_values = constraints.get('unique_values')
        if unique_values:
            return random.choice(unique_values)
        
        # Get range constraints, with fallback to data analysis
        min_val = constraints.get('min_value')
        max_val = constraints.get('max_value')
        is_integer = constraints.get('is_integer', False)
        
        # Fallback: analyze data directly if schema missing values
        if (min_val is None or max_val is None) and col_name in reference_data.columns:
            col_data = reference_data[col_name].dropna()
            if len(col_data) > 0:
                min_val = min_val or float(col_data.min())
                max_val = max_val or float(col_data.max())
                is_integer = is_integer or pd.api.types.is_integer_dtype(col_data)
        
        # Set defaults if still None
        min_val = min_val or 0
        max_val = max_val or 100
        
        # Try to generate around typical range if available
        quartiles = distribution_info.get('quartiles', {})
        if quartiles:
            q25 = quartiles.get('q25', min_val)
            q75 = quartiles.get('q75', max_val)
            
            # 70% chance to generate in interquartile range, 30% in full range
            if random.random() < 0.7:
                min_val, max_val = max(min_val, q25), min(max_val, q75)
        
        # Generate value
        try:
            if is_integer:
                return random.randint(int(min_val), int(max_val))
            else:
                return round(random.uniform(min_val, max_val), 2)
        except ValueError:
            # Fallback if ranges are invalid
            return 0 if is_integer else 0.0

    def _generate_datetime_value(self, col_schema: Dict, reference_data: pd.DataFrame, col_name: str) -> str:
        """Generate a datetime value based on schema constraints."""
        
        constraints = col_schema.get('constraints', {})
        min_date = constraints.get('min_date')
        max_date = constraints.get('max_date')
        formats = constraints.get('common_formats', ['%Y-%m-%d'])
        
        # Use faker to generate date in range
        if min_date and max_date:
            try:
                start_date = pd.to_datetime(min_date).date()
                end_date = pd.to_datetime(max_date).date()
                fake_date = self.faker.date_between(start_date=start_date, end_date=end_date)
            except:
                fake_date = self.faker.date_of_birth(minimum_age=18, maximum_age=25)
        else:
            fake_date = self.faker.date_of_birth(minimum_age=18, maximum_age=25)
        
        # Format according to detected format
        if formats:
            try:
                format_str = formats[0]
                if '%H:%M:%S' in format_str:
                    # Add time component
                    fake_datetime = self.faker.date_time_between(start_date=fake_date, end_date=fake_date + pd.Timedelta(days=1))
                    return fake_datetime.strftime(format_str)
                else:
                    return fake_date.strftime(format_str)
            except:
                return fake_date.isoformat()
        
        return fake_date.isoformat()

    def _generate_text_value(self, col_schema: Dict, reference_data: pd.DataFrame, col_name: str) -> str:
        """Generate a text value based on schema constraints."""
        
        constraints = col_schema.get('constraints', {})
        
        # Check if limited possible values
        possible_values = constraints.get('possible_values')
        if not possible_values and col_name in reference_data.columns:
            # Fallback: if few unique values, treat as categorical
            unique_vals = reference_data[col_name].dropna().unique()
            if len(unique_vals) <= 50:  # Reasonable threshold
                possible_values = unique_vals.tolist()
        
        if possible_values:
            return random.choice(possible_values)
        
        # Generate based on length constraints
        min_length = constraints.get('min_length', 3)
        max_length = constraints.get('max_length', 50)
        patterns = constraints.get('common_patterns', [])
        
        # Generate based on detected patterns
        if 'code_pattern' in patterns:
            return f"{random.choice(['STU', 'ID', 'CODE'])}{random.randint(100, 9999)}"
        elif 'email_like' in patterns:
            return self.faker.email()
        elif 'capitalized_word' in patterns:
            return self.faker.word().capitalize()
        elif 'numeric_string' in patterns:
            return str(random.randint(10**(min_length-1), 10**max_length - 1))
        else:
            # Generate random text within length constraints
            target_length = random.randint(max(1, min_length), min(max_length, 50))
            return self.faker.text(max_nb_chars=target_length).replace('\n', ' ').strip()

    def _generate_fallback_value(self, col_name: str, reference_data: pd.DataFrame) -> Any:
        """Generate fallback value when schema extraction fails."""
        
        if col_name in reference_data.columns:
            # Sample from existing values
            non_null_values = reference_data[col_name].dropna()
            if len(non_null_values) > 0:
                return random.choice(non_null_values.tolist())
        
        # Ultimate fallback
        return 'Unknown'

    # Keep remaining methods for compatibility
    def generate_llm_prompt(self, dataset_name: str, n_samples: int, sample_data: Dict) -> str:
        return ""

    def generate_with_llm(self, dataset_name: str, n_samples: int, sample_data: Dict) -> List[Dict]:
        return []

    def enhance_with_faker(self, data: List[Dict], dataset_name: str) -> List[Dict]:
        return data

    def balance_groups(self, data: List[Dict], dataset_name: str, 
                      sensitive_ratio: float, label_ratio: float) -> List[Dict]:
        return data

    def generate_synthetic_dataset(self, dataset_name: str, n_samples: int,
                                 sensitive_ratio: float = 0.5, label_ratio: float = 0.5,
                                 reference_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return pd.DataFrame()

    def _get_default_sample(self, dataset_name: str) -> List[Dict]:
        return [{}]

    def _generate_faker_fallback(self, dataset_name: str, n_samples: int) -> List[Dict]:
        return []