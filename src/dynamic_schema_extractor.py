# ModularCode/src/dynamic_schema_extractor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime
import re

class DynamicSchemaExtractor:
    """
    Automatically extracts schema and constraints from datasets to ensure 
    synthetic data generation stays within the original data distribution.
    """
    
    def __init__(self, max_unique_threshold: int = 50, sample_size_for_analysis: int = 1000):
        """
        Initialize the schema extractor.
        
        Parameters:
        max_unique_threshold: Maximum unique values to treat as categorical
        sample_size_for_analysis: Sample size for analyzing large datasets
        """
        self.max_unique_threshold = max_unique_threshold
        self.sample_size_for_analysis = sample_size_for_analysis
        
    def extract_comprehensive_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract a comprehensive schema from the dataset including:
        - Column types and constraints
        - Value distributions
        - Relationships between columns
        - Statistical properties
        """
        
        # Sample data if too large for efficient analysis
        if len(data) > self.sample_size_for_analysis:
            analysis_data = data.sample(n=self.sample_size_for_analysis, random_state=42)
            print(f"Analyzing schema from {self.sample_size_for_analysis} samples (dataset has {len(data)} total)")
        else:
            analysis_data = data.copy()
        
        schema = {
            'dataset_info': {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'analysis_sample_size': len(analysis_data),
                'extraction_timestamp': datetime.now().isoformat()
            },
            'columns': {},
            'relationships': {},
            'global_constraints': {}
        }
        
        # Extract column-specific schemas
        for column in data.columns:
            schema['columns'][column] = self._extract_column_schema(analysis_data[column], column)
        
        # Extract relationships between key columns (sensitive and label columns)
        schema['relationships'] = self._extract_column_relationships(analysis_data, schema['columns'])
        
        # Extract global constraints
        schema['global_constraints'] = self._extract_global_constraints(analysis_data, schema['columns'])
        
        return schema
    
    def _extract_column_schema(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Extract detailed schema for a single column."""
        
        column_schema = {
            'name': column_name,
            'dtype': str(series.dtype),
            'total_count': len(series),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'type_category': None,
            'constraints': {},
            'distribution_info': {}
        }
        
        # Remove nulls for analysis
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            column_schema['type_category'] = 'empty'
            return column_schema
        
        # Determine column type category and extract constraints
        if self._is_datetime_column(clean_series):
            column_schema.update(self._extract_datetime_constraints(clean_series))
        elif self._is_numeric_column(clean_series):
            column_schema.update(self._extract_numeric_constraints(clean_series))
        elif self._is_categorical_column(clean_series):
            column_schema.update(self._extract_categorical_constraints(clean_series))
        else:
            column_schema.update(self._extract_text_constraints(clean_series))
        
        return column_schema
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Check if column contains datetime values."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Try to parse a sample as datetime
        sample_values = series.head(10).astype(str)
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        ]
        
        for value in sample_values:
            for pattern in datetime_patterns:
                if re.search(pattern, str(value)):
                    return True
        return False
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if column is numeric."""
        return pd.api.types.is_numeric_dtype(series)
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Check if column should be treated as categorical."""
        unique_count = series.nunique()
        total_count = len(series)
        
        # Treat as categorical if:
        # 1. Few unique values relative to total
        # 2. String/object type with reasonable unique count
        # 3. Boolean type
        
        if pd.api.types.is_bool_dtype(series):
            return True
        
        if unique_count <= self.max_unique_threshold:
            return True
        
        if pd.api.types.is_object_dtype(series) and unique_count / total_count < 0.5:
            return True
        
        return False
    
    def _extract_datetime_constraints(self, series: pd.Series) -> Dict[str, Any]:
        """Extract constraints for datetime columns."""
        try:
            # Try to convert to datetime
            if not pd.api.types.is_datetime64_any_dtype(series):
                dt_series = pd.to_datetime(series, errors='coerce').dropna()
            else:
                dt_series = series
            
            if len(dt_series) == 0:
                return {'type_category': 'text'}  # Fall back to text if can't parse
            
            return {
                'type_category': 'datetime',
                'constraints': {
                    'min_date': dt_series.min().isoformat(),
                    'max_date': dt_series.max().isoformat(),
                    'date_range_days': (dt_series.max() - dt_series.min()).days,
                    'common_formats': self._detect_datetime_formats(series.head(20))
                },
                'distribution_info': {
                    'year_range': [dt_series.dt.year.min(), dt_series.dt.year.max()],
                    'month_distribution': dt_series.dt.month.value_counts().to_dict(),
                    'weekday_distribution': dt_series.dt.dayofweek.value_counts().to_dict()
                }
            }
        except Exception:
            return {'type_category': 'text'}  # Fall back to text
    
    def _extract_numeric_constraints(self, series: pd.Series) -> Dict[str, Any]:
        """Extract constraints for numeric columns."""
        
        # Determine if integer or float
        is_integer = pd.api.types.is_integer_dtype(series) or series.apply(lambda x: float(x).is_integer()).all()
        
        constraints = {
            'min_value': float(series.min()),
            'max_value': float(series.max()),
            'is_integer': is_integer,
            'range': float(series.max() - series.min())
        }
        
        # Add integer-specific constraints
        if is_integer:
            constraints['unique_values'] = sorted(series.unique().tolist()) if series.nunique() <= self.max_unique_threshold else None
        
        distribution_info = {
            'mean': float(series.mean()),
            'median': float(series.median()),
            'std': float(series.std()),
            'quartiles': {
                'q05': float(series.quantile(0.05)),  # NEW: 5th percentile for robust lower bound
                'q25': float(series.quantile(0.25)),
                'q50': float(series.quantile(0.50)),
                'q75': float(series.quantile(0.75)),
                'q95': float(series.quantile(0.95))   # NEW: 95th percentile for robust upper bound
            },
            'outliers_count': self._count_outliers(series),
            'distribution_type': self._detect_distribution_type(series)
        }
        
        return {
            'type_category': 'numeric',
            'constraints': constraints,
            'distribution_info': distribution_info
        }
    
    def _extract_categorical_constraints(self, series: pd.Series) -> Dict[str, Any]:
        """Extract constraints for categorical columns."""
        
        value_counts = series.value_counts()
        unique_values = series.unique().tolist()
        
        # Sort unique values for consistency
        try:
            unique_values = sorted(unique_values)
        except TypeError:
            # Mixed types, keep original order
            pass
        
        constraints = {
            'allowed_values': unique_values,
            'value_counts': value_counts.to_dict(),
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'least_common': value_counts.index[-1] if len(value_counts) > 0 else None
        }
        
        distribution_info = {
            'unique_count': len(unique_values),
            'entropy': self._calculate_entropy(value_counts),
            'dominance_ratio': value_counts.iloc[0] / len(series) if len(value_counts) > 0 else 0,
            'is_binary': len(unique_values) == 2,
            'is_boolean_like': self._is_boolean_like(unique_values)
        }
        
        return {
            'type_category': 'categorical',
            'constraints': constraints,
            'distribution_info': distribution_info
        }
    
    def _extract_text_constraints(self, series: pd.Series) -> Dict[str, Any]:
        """Extract constraints for text columns."""
        
        str_series = series.astype(str)
        
        constraints = {
            'min_length': str_series.str.len().min(),
            'max_length': str_series.str.len().max(),
            'avg_length': str_series.str.len().mean(),
            'common_patterns': self._extract_text_patterns(str_series),
            'character_set': self._extract_character_set(str_series)
        }
        
        # If few unique values, treat somewhat like categorical
        if series.nunique() <= self.max_unique_threshold:
            constraints['possible_values'] = series.unique().tolist()
        
        distribution_info = {
            'length_distribution': str_series.str.len().value_counts().head(10).to_dict(),
            'starts_with_patterns': self._extract_prefix_patterns(str_series),
            'contains_numbers': str_series.str.contains(r'\d').sum(),
            'contains_special_chars': str_series.str.contains(r'[^a-zA-Z0-9\s]').sum()
        }
        
        return {
            'type_category': 'text',
            'constraints': constraints,
            'distribution_info': distribution_info
        }
    
    def _extract_column_relationships(self, data: pd.DataFrame, column_schemas: Dict) -> Dict[str, Any]:
        """Extract relationships between columns, especially for sensitive and label columns."""
        
        relationships = {
            'correlations': {},
            'cross_tabulations': {},
            'conditional_distributions': {}
        }
        
        # Find potential sensitive and label columns
        sensitive_candidates = self._find_sensitive_column_candidates(data, column_schemas)
        label_candidates = self._find_label_column_candidates(data, column_schemas)
        
        relationships['sensitive_column_candidates'] = sensitive_candidates
        relationships['label_column_candidates'] = label_candidates
        
        # Extract cross-tabulations for categorical pairs
        categorical_columns = [col for col, schema in column_schemas.items() 
                             if schema.get('type_category') == 'categorical']
        
        for i, col1 in enumerate(categorical_columns):
            for col2 in categorical_columns[i+1:]:
                cross_tab = pd.crosstab(data[col1], data[col2])
                relationships['cross_tabulations'][f"{col1}_vs_{col2}"] = cross_tab.to_dict()
        
        # Extract correlations for numeric columns
        numeric_columns = [col for col, schema in column_schemas.items() 
                          if schema.get('type_category') == 'numeric']
        
        if len(numeric_columns) > 1:
            corr_matrix = data[numeric_columns].corr()
            relationships['correlations'] = corr_matrix.to_dict()
        
        return relationships
    
    def _extract_global_constraints(self, data: pd.DataFrame, column_schemas: Dict) -> Dict[str, Any]:
        """Extract global constraints that apply across columns."""
        
        return {
            'total_row_constraints': {
                'typical_row_count': len(data),
                'column_completion_rates': {col: (1 - schema['null_percentage']/100) 
                                          for col, schema in column_schemas.items()}
            },
            'data_quality_indicators': {
                'overall_completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))),
                'columns_with_high_nulls': [col for col, schema in column_schemas.items() 
                                          if schema['null_percentage'] > 50],
                'uniform_columns': [col for col, schema in column_schemas.items() 
                                  if schema.get('type_category') == 'categorical' and 
                                  schema.get('distribution_info', {}).get('unique_count', 0) == 1]
            }
        }
    
    # Helper methods
    def _detect_datetime_formats(self, sample_series: pd.Series) -> List[str]:
        """Detect common datetime formats in the data."""
        formats = []
        str_sample = sample_series.astype(str).head(10)
        
        common_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f'
        ]
        
        for fmt in common_formats:
            try:
                parsed_count = 0
                for val in str_sample:
                    try:
                        datetime.strptime(str(val), fmt)
                        parsed_count += 1
                    except:
                        continue
                if parsed_count > len(str_sample) * 0.5:  # If >50% match
                    formats.append(fmt)
            except:
                continue
        
        return formats
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def _detect_distribution_type(self, series: pd.Series) -> str:
        """Detect the likely distribution type of numeric data."""
        # Simple heuristics for common distributions
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return 'normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        elif kurtosis > 3:
            return 'heavy_tailed'
        else:
            return 'unknown'
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution."""
        probabilities = value_counts / value_counts.sum()
        return -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    def _is_boolean_like(self, unique_values: List) -> bool:
        """Check if values are boolean-like."""
        if len(unique_values) != 2:
            return False
        
        str_values = [str(v).lower() for v in unique_values]
        boolean_pairs = [
            {'true', 'false'}, {'yes', 'no'}, {'1', '0'}, 
            {'male', 'female'}, {'m', 'f'}, {'dropout', 'graduate'},
            {'dropout', 'not dropout'}, {'positive', 'negative'}
        ]
        
        return set(str_values) in boolean_pairs or any(
            set(str_values).issubset(pair) for pair in boolean_pairs
        )
    
    def _extract_text_patterns(self, str_series: pd.Series) -> List[str]:
        """Extract common text patterns."""
        patterns = []
        sample = str_series.head(20)
        
        # Common patterns
        if sample.str.match(r'^[A-Z][a-z]+$').any():
            patterns.append('capitalized_word')
        if sample.str.match(r'^\d+$').any():
            patterns.append('numeric_string')
        if sample.str.match(r'^[A-Z]{1,5}\d+$').any():
            patterns.append('code_pattern')
        if sample.str.contains(r'@').any():
            patterns.append('email_like')
        
        return patterns
    
    def _extract_character_set(self, str_series: pd.Series) -> str:
        """Determine the character set used."""
        all_chars = set(''.join(str_series.head(100).astype(str)))
        
        if all_chars.issubset(set('0123456789')):
            return 'numeric'
        elif all_chars.issubset(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')):
            return 'alphabetic'
        elif all_chars.issubset(set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ')):
            return 'alphanumeric'
        else:
            return 'mixed'
    
    def _extract_prefix_patterns(self, str_series: pd.Series) -> Dict[str, int]:
        """Extract common prefix patterns."""
        prefixes = str_series.str[:3].value_counts().head(5)
        return prefixes.to_dict()
    
    def _find_sensitive_column_candidates(self, data: pd.DataFrame, column_schemas: Dict) -> List[str]:
        """Find likely sensitive attribute columns."""
        candidates = []
        
        for col, schema in column_schemas.items():
            col_lower = col.lower()
            
            # Check for common sensitive attribute names
            if any(keyword in col_lower for keyword in ['gender', 'sex', 'race', 'ethnicity', 'age']):
                candidates.append(col)
            
            # Check for binary categorical that might be sensitive
            elif (schema.get('type_category') == 'categorical' and 
                  schema.get('distribution_info', {}).get('is_binary', False)):
                candidates.append(col)
        
        return candidates
    
    def _find_label_column_candidates(self, data: pd.DataFrame, column_schemas: Dict) -> List[str]:
        """Find likely label/target columns."""
        candidates = []
        
        for col, schema in column_schemas.items():
            col_lower = col.lower()
            
            # Check for common target names
            if any(keyword in col_lower for keyword in ['target', 'label', 'outcome', 'dropout', 'class', 'result']):
                candidates.append(col)
            
            # Check for binary categorical that might be target
            elif (schema.get('type_category') == 'categorical' and 
                  schema.get('distribution_info', {}).get('is_binary', False)):
                candidates.append(col)
        
        return candidates

    def create_generation_prompt_schema(self, schema: Dict[str, Any], 
                                      target_columns: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a concise schema description optimized for LLM prompt generation.
        """
        
        prompt_schema = {}
        
        for col_name, col_schema in schema['columns'].items():
            type_category = col_schema.get('type_category', 'unknown')
            
            if type_category == 'categorical':
                allowed_values = col_schema.get('constraints', {}).get('allowed_values', [])
                prompt_schema[col_name] = {
                    'type': 'categorical',
                    'allowed_values': allowed_values,
                    'most_common': col_schema.get('constraints', {}).get('most_common')
                }
            
            elif type_category == 'numeric':
                constraints = col_schema.get('constraints', {})
                prompt_schema[col_name] = {
                    'type': 'integer' if constraints.get('is_integer', False) else 'float',
                    'min': constraints.get('min_value'),
                    'max': constraints.get('max_value'),
                    'typical_range': [
                        col_schema.get('distribution_info', {}).get('quartiles', {}).get('q25'),
                        col_schema.get('distribution_info', {}).get('quartiles', {}).get('q75')
                    ]
                }
                
                # Add specific values if few unique values
                if constraints.get('unique_values'):
                    prompt_schema[col_name]['allowed_values'] = constraints['unique_values']
            
            elif type_category == 'datetime':
                constraints = col_schema.get('constraints', {})
                prompt_schema[col_name] = {
                    'type': 'datetime',
                    'min_date': constraints.get('min_date'),
                    'max_date': constraints.get('max_date'),
                    'formats': constraints.get('common_formats', ['%Y-%m-%d'])
                }
            
            elif type_category == 'text':
                constraints = col_schema.get('constraints', {})
                prompt_schema[col_name] = {
                    'type': 'text',
                    'min_length': constraints.get('min_length'),
                    'max_length': constraints.get('max_length'),
                    'patterns': constraints.get('common_patterns', [])
                }
                
                # Add possible values if limited
                if constraints.get('possible_values'):
                    prompt_schema[col_name]['allowed_values'] = constraints['possible_values']
        
        # Add target column constraints if specified
        if target_columns:
            for col_name, target_value in target_columns.items():
                if col_name in prompt_schema:
                    prompt_schema[col_name]['target_value'] = target_value
        
        return json.dumps(prompt_schema, indent=2, default=str)