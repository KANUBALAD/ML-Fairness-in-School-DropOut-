# ModularCode/src/llm_synthetic_generator.py (v11 - Complete Multi-Provider Support)

import requests
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .dynamic_schema_extractor import DynamicSchemaExtractor

class ThreadedLLMSyntheticGenerator:
    """
    Universal LLM generator supporting multiple providers with generic model specification:
    - DeepSeek API
    - Hugging Face Inference API  
    - Hugging Face Transformers (local)
    - OpenAI API (future)
    - Any OpenAI-compatible API
    """
    def __init__(self, api_key: str, provider: str = "deepseek", model_name: str = None, 
                 base_url: str = None, max_workers: int = 6):
        if not api_key or api_key == "dummy":
            if provider != "huggingface_transformers":
                raise ValueError("A valid API key is required for API-based providers.")
        
        self.api_key = api_key
        self.provider = provider.lower()
        self.max_workers = max_workers
        
        # Configure provider-specific settings
        self._configure_provider(model_name, base_url)
        
        # Initialize dynamic schema extractor
        self.schema_extractor = DynamicSchemaExtractor()
        self._dataset_schemas = {}
        self._generated_samples_cache = []
        self._diversity_stats = {}
        
        print(f"🚀 LLM Generator initialized: {self.provider.upper()} with {max_workers} workers")
        print(f"   Model: {self.model_name}")

    def _configure_provider(self, model_name: str = None, base_url: str = None):
        """Configure provider-specific settings with generic model support."""
        
        if self.provider == "deepseek":
            self.base_url = base_url or "https://api.deepseek.com/v1"
            self.model_name = model_name or "deepseek-chat"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.api_type = "openai_compatible"
            
        elif self.provider == "huggingface":
            self.base_url = "https://api-inference.huggingface.co/models"
            # Default to a reliable model if none specified
            self.model_name = model_name or "microsoft/DialoGPT-medium"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.api_type = "huggingface_inference"
            
        elif self.provider == "huggingface_transformers":
            # For local transformers usage
            self.model_name = model_name or "microsoft/DialoGPT-medium"
            self.api_type = "transformers_local"
            self._initialize_local_model()
            
        elif self.provider == "openai":
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model_name = model_name or "gpt-3.5-turbo"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.api_type = "openai_compatible"
            
        elif self.provider == "custom_openai":
            # For any OpenAI-compatible API
            if not base_url:
                raise ValueError("base_url is required for custom_openai provider")
            self.base_url = base_url
            self.model_name = model_name or "default"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self.api_type = "openai_compatible"
            
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Supported: deepseek, huggingface, huggingface_transformers, openai, custom_openai")

    def _initialize_local_model(self):
        """Initialize local Hugging Face transformers model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            print(f"📥 Loading model {self.model_name} locally...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # Use CPU (change to 0 for GPU)
                return_full_text=False
            )
            print(f"✅ Model loaded successfully")
            
        except ImportError:
            raise ImportError("transformers library required for local model usage. Install with: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}")

    def _get_or_extract_schema(self, dataset_name: str, reference_data: pd.DataFrame) -> Dict:
        """Get cached schema or extract new one with enhanced analysis."""
        
        if dataset_name not in self._dataset_schemas:
            print(f"Extracting dynamic schema for dataset: {dataset_name}")
            schema = self.schema_extractor.extract_comprehensive_schema(reference_data)
            
            # Add universal diversity enhancement metadata
            schema['diversity_metadata'] = self._extract_universal_diversity_metadata(reference_data, schema)
            
            # Add universal logical relationships
            schema['logical_relationships'] = self._extract_universal_logical_relationships(reference_data, schema)
            
            self._dataset_schemas[dataset_name] = schema
            print(f"✓ Schema extracted with {len(schema['columns'])} columns analyzed")
        
        return self._dataset_schemas[dataset_name]

    def _extract_universal_diversity_metadata(self, reference_data: pd.DataFrame, schema: Dict) -> Dict:
        """Extract diversity metadata that works for any dataset structure."""
        
        diversity_info = {
            'column_diversity_levels': {},
            'value_distribution_analysis': {},
            'statistical_properties': {}
        }
        
        for col_name, col_schema in schema['columns'].items():
            if col_name not in reference_data.columns:
                continue
                
            type_category = col_schema.get('type_category', 'unknown')
            col_data = reference_data[col_name].dropna()
            
            if len(col_data) == 0:
                continue
            
            if type_category == 'categorical':
                # Universal categorical analysis
                value_counts = col_data.value_counts()
                total_count = len(col_data)
                
                # Calculate distribution characteristics
                entropy = self._calculate_entropy(value_counts)
                gini_coefficient = self._calculate_gini_coefficient(value_counts)
                
                # Identify value distribution without hardcoding specific values
                sorted_counts = value_counts.sort_values(ascending=False)
                dominant_threshold = total_count * 0.5  # Values that dominate (>50%)
                rare_threshold = total_count * 0.05     # Values that are rare (<5%)
                
                dominant_values = sorted_counts[sorted_counts >= dominant_threshold].index.tolist()
                rare_values = sorted_counts[sorted_counts <= rare_threshold].index.tolist()
                moderate_values = sorted_counts[(sorted_counts > rare_threshold) & (sorted_counts < dominant_threshold)].index.tolist()
                
                diversity_info['value_distribution_analysis'][col_name] = {
                    'dominant_values': dominant_values,
                    'moderate_values': moderate_values,
                    'rare_values': rare_values,
                    'entropy': entropy,
                    'gini_coefficient': gini_coefficient,
                    'uniformity_score': entropy / np.log2(len(value_counts)) if len(value_counts) > 1 else 0
                }
                
            elif type_category == 'numeric':
                # Universal numeric analysis
                diversity_info['statistical_properties'][col_name] = {
                    'distribution_shape': self._analyze_distribution_shape(col_data),
                    'variability_level': self._calculate_variability_level(col_data),
                    'outlier_characteristics': self._analyze_outliers(col_data),
                    'range_utilization': self._calculate_range_utilization(col_data)
                }
        
        return diversity_info

    def _extract_universal_logical_relationships(self, reference_data: pd.DataFrame, schema: Dict) -> Dict:
        """Extract logical relationships without dataset-specific assumptions."""
        
        relationships = {
            'numeric_constraints': [],
            'categorical_dependencies': [],
            'hierarchical_patterns': []
        }
        
        # Analyze numeric columns for potential constraints
        numeric_cols = [col for col, col_schema in schema['columns'].items() 
                       if col_schema.get('type_category') == 'numeric']
        
        # Find potential min/max or part/whole relationships using correlation and naming
        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i+1:]:
                relationship = self._detect_universal_numeric_relationship(reference_data, col_a, col_b)
                if relationship:
                    relationships['numeric_constraints'].append(relationship)
        
        # Analyze categorical dependencies
        categorical_cols = [col for col, col_schema in schema['columns'].items() 
                           if col_schema.get('type_category') == 'categorical']
        
        for i, col_a in enumerate(categorical_cols):
            for col_b in categorical_cols[i+1:]:
                dependency = self._detect_categorical_dependency(reference_data, col_a, col_b)
                if dependency:
                    relationships['categorical_dependencies'].append(dependency)
        
        return relationships

    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy for diversity measurement."""
        probabilities = value_counts / value_counts.sum()
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return float(entropy)

    def _calculate_gini_coefficient(self, value_counts: pd.Series) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        sorted_counts = np.sort(value_counts.values)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0

    def _analyze_distribution_shape(self, series: pd.Series) -> str:
        """Analyze the shape of numeric distribution."""
        skewness = series.skew()
        
        if abs(skewness) < 0.5:
            return 'symmetric'
        elif skewness > 0.5:
            return 'right_skewed'
        elif skewness < -0.5:
            return 'left_skewed'
        else:
            return 'moderate_skew'

    def _calculate_variability_level(self, series: pd.Series) -> str:
        """Calculate variability level of numeric data."""
        cv = series.std() / series.mean() if series.mean() != 0 else 0
        
        if cv < 0.1:
            return 'low'
        elif cv < 0.5:
            return 'moderate'
        else:
            return 'high'

    def _analyze_outliers(self, series: pd.Series) -> Dict:
        """Analyze outlier characteristics."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(series) * 100,
            'outlier_severity': 'high' if len(outliers) / len(series) > 0.1 else 'low'
        }

    def _calculate_range_utilization(self, series: pd.Series) -> float:
        """Calculate how much of the theoretical range is utilized."""
        actual_range = series.max() - series.min()
        # Use IQR as a measure of typical variation
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        typical_range = Q3 - Q1
        
        return typical_range / actual_range if actual_range > 0 else 1.0

    def _detect_universal_numeric_relationship(self, data: pd.DataFrame, col_a: str, col_b: str) -> Optional[Dict]:
        """Detect numeric relationships without hardcoded patterns."""
        
        try:
            # Skip if either column has too many nulls
            if data[col_a].isnull().sum() / len(data) > 0.5 or data[col_b].isnull().sum() / len(data) > 0.5:
                return None
            
            clean_data = data[[col_a, col_b]].dropna()
            if len(clean_data) < 10:
                return None
            
            # Check for potential hierarchical relationship (one always >= other)
            a_ge_b_rate = (clean_data[col_a] >= clean_data[col_b]).mean()
            b_ge_a_rate = (clean_data[col_b] >= clean_data[col_a]).mean()
            
            # If one column is consistently >= the other (>90% of time)
            if a_ge_b_rate > 0.9:
                return {
                    'type': 'hierarchical_numeric',
                    'higher_col': col_a,
                    'lower_col': col_b,
                    'consistency_rate': a_ge_b_rate,
                    'constraint': f"{col_a} >= {col_b}"
                }
            elif b_ge_a_rate > 0.9:
                return {
                    'type': 'hierarchical_numeric',
                    'higher_col': col_b,
                    'lower_col': col_a,
                    'consistency_rate': b_ge_a_rate,
                    'constraint': f"{col_b} >= {col_a}"
                }
            
            # Check for strong correlations
            correlation = clean_data[col_a].corr(clean_data[col_b])
            if abs(correlation) > 0.8:
                return {
                    'type': 'strong_correlation',
                    'col_a': col_a,
                    'col_b': col_b,
                    'correlation': correlation,
                    'relationship_strength': 'strong' if abs(correlation) > 0.9 else 'moderate'
                }
            
            return None
            
        except Exception:
            return None

    def _detect_categorical_dependency(self, data: pd.DataFrame, col_a: str, col_b: str) -> Optional[Dict]:
        """Detect categorical dependencies using statistical measures."""
        
        try:
            # Create contingency table
            contingency = pd.crosstab(data[col_a], data[col_b])
            
            # Calculate Cramér's V for association strength
            chi2 = self._calculate_chi_square(contingency)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            # Strong dependency if Cramér's V > 0.5
            if cramers_v > 0.5:
                return {
                    'type': 'categorical_dependency',
                    'col_a': col_a,
                    'col_b': col_b,
                    'association_strength': cramers_v,
                    'dependency_level': 'strong' if cramers_v > 0.7 else 'moderate'
                }
            
            return None
            
        except Exception:
            return None

    def _calculate_chi_square(self, contingency_table: pd.DataFrame) -> float:
        """Calculate chi-square statistic for contingency table."""
        try:
            from scipy.stats import chi2_contingency
            chi2, _, _, _ = chi2_contingency(contingency_table)
            return chi2
        except:
            # Fallback calculation if scipy not available
            expected = np.outer(contingency_table.sum(axis=1), contingency_table.sum(axis=0)) / contingency_table.sum().sum()
            chi2 = ((contingency_table - expected) ** 2 / expected).sum().sum()
            return float(chi2)

    def _identify_sensitive_and_label_columns(self, dataset_name: str, schema: Dict, reference_data: pd.DataFrame) -> Dict[str, str]:
        """Automatically identify sensitive and label columns using universal heuristics."""
        
        relationships = schema.get('relationships', {})
        sensitive_candidates = relationships.get('sensitive_column_candidates', [])
        label_candidates = relationships.get('label_column_candidates', [])
        
        # Universal heuristics for sensitive attributes
        sensitive_col = None
        if sensitive_candidates:
            # Look for binary categorical columns that might represent protected attributes
            for candidate in sensitive_candidates:
                col_schema = schema['columns'].get(candidate, {})
                if col_schema.get('type_category') == 'categorical':
                    constraints = col_schema.get('constraints', {})
                    allowed_values = constraints.get('allowed_values', [])
                    
                    # Prefer binary columns with balanced distribution
                    if len(allowed_values) == 2:
                        value_counts = constraints.get('value_counts', {})
                        if value_counts:
                            balance_ratio = min(value_counts.values()) / max(value_counts.values())
                            if balance_ratio > 0.2:  # Not too imbalanced
                                sensitive_col = candidate
                                break
            
            if not sensitive_col:
                sensitive_col = sensitive_candidates[0]  # Take first candidate
        
        # Universal heuristics for label columns
        label_col = None
        if label_candidates:
            # Look for columns that represent outcomes
            for candidate in label_candidates:
                col_schema = schema['columns'].get(candidate, {})
                if col_schema.get('type_category') == 'categorical':
                    constraints = col_schema.get('constraints', {})
                    allowed_values = constraints.get('allowed_values', [])
                    
                    # Prefer binary outcome columns
                    if len(allowed_values) == 2:
                        label_col = candidate
                        break
            
            if not label_col:
                label_col = label_candidates[0]  # Take first candidate

        # Fallback: automatic detection based on column properties
        if not sensitive_col:
            sensitive_col = self._find_best_sensitive_column(reference_data, schema)

        if not label_col:
            label_col = self._find_best_label_column(reference_data, schema)
        
        if not sensitive_col or not label_col:
            available_cols = list(reference_data.columns)
            raise ValueError(f"Could not automatically identify key columns. Available columns: {available_cols}")
        
        print(f"Auto-identified columns: sensitive='{sensitive_col}', label='{label_col}'")
        return {'sensitive_col': sensitive_col, 'label_col': label_col}

    def _find_best_sensitive_column(self, reference_data: pd.DataFrame, schema: Dict) -> Optional[str]:
        """Find the best sensitive attribute column using universal criteria."""
        
        candidates = []
        
        for col_name, col_schema in schema['columns'].items():
            if col_schema.get('type_category') == 'categorical':
                constraints = col_schema.get('constraints', {})
                allowed_values = constraints.get('allowed_values', [])
                
                # Score potential sensitive attributes
                score = 0
                
                # Prefer binary columns
                if len(allowed_values) == 2:
                    score += 50
                
                # Prefer balanced distributions
                value_counts = constraints.get('value_counts', {})
                if value_counts and len(value_counts) >= 2:
                    values = list(value_counts.values())
                    balance_ratio = min(values) / max(values)
                    score += balance_ratio * 30
                
                # Prefer columns with reasonable entropy
                diversity_info = schema.get('diversity_metadata', {}).get('value_distribution_analysis', {}).get(col_name, {})
                entropy = diversity_info.get('entropy', 0)
                if 0.5 < entropy < 2.0:  # Not too uniform, not too chaotic
                    score += 20
                
                candidates.append((col_name, score))
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[1])
            if best_candidate[1] > 40:  # Minimum threshold
                return best_candidate[0]
        
        return None

    def _find_best_label_column(self, reference_data: pd.DataFrame, schema: Dict) -> Optional[str]:
        """Find the best label column using universal criteria."""
        
        candidates = []
        
        for col_name, col_schema in schema['columns'].items():
            if col_schema.get('type_category') == 'categorical':
                constraints = col_schema.get('constraints', {})
                allowed_values = constraints.get('allowed_values', [])
                
                # Score potential label columns
                score = 0
                
                # Prefer binary outcomes
                if len(allowed_values) == 2:
                    score += 40
                elif len(allowed_values) <= 5:  # Multi-class is okay too
                    score += 20
                
                # Prefer columns that could represent outcomes (typically imbalanced)
                value_counts = constraints.get('value_counts', {})
                if value_counts and len(value_counts) >= 2:
                    values = list(value_counts.values())
                    imbalance_ratio = min(values) / max(values)
                    if 0.1 < imbalance_ratio < 0.8:  # Some imbalance suggests outcome variable
                        score += 30
                
                candidates.append((col_name, score))
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[1])
            if best_candidate[1] > 30:  # Minimum threshold
                return best_candidate[0]
        
        return None

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
            return unique_values
        
        raise ValueError(f"Could not extract values for column '{col_name}'")

    def _get_column_values_for_category(self, schema: Dict, columns: Dict[str, str], category: str, reference_data: pd.DataFrame) -> Tuple:
        """Get the specific values for sensitive and label columns based on category."""
        
        sensitive_col = columns['sensitive_col']
        label_col = columns['label_col']
        
        # Safely get values for both columns
        try:
            sensitive_values = self._get_column_values_safely(schema, reference_data, sensitive_col)
            label_values = self._get_column_values_safely(schema, reference_data, label_col)
        except Exception as e:
            print(f"  Error extracting column values: {e}")
            raise
        
        # Determine privileged/unprivileged and positive/negative values using heuristics
        privileged_val, unprivileged_val = self._determine_privileged_values(sensitive_values, sensitive_col)
        positive_val, negative_val = self._determine_positive_values(label_values, label_col)
        
        # Map category to specific values
        category_mapping = {
            'priv_pos': (privileged_val, positive_val),
            'priv_neg': (privileged_val, negative_val), 
            'unpriv_pos': (unprivileged_val, positive_val),
            'unpriv_neg': (unprivileged_val, negative_val)
        }
        
        if category not in category_mapping:
            raise ValueError(f"Unknown category: {category}")
        
        return category_mapping[category]

    def _determine_privileged_values(self, values: List, column_name: str) -> Tuple:
        """Determine which value represents privileged vs unprivileged group using universal heuristics."""
        
        if len(values) < 2:
            return values[0] if values else 'Unknown', values[1] if len(values) > 1 else 'Unknown'
        
        val1, val2 = values[0], values[1]
        col_lower = column_name.lower()
        
        # Look for common patterns in column names and values
        val1_str, val2_str = str(val1).lower(), str(val2).lower()
        
        # Gender/sex patterns
        if any(word in col_lower for word in ['gender', 'sex']):
            # Male-related values typically considered privileged in fairness literature
            male_indicators = ['male', 'm', '1', 'man', 'masculine']
            female_indicators = ['female', 'f', '0', 'woman', 'feminine']
            
            if any(indicator in val1_str for indicator in male_indicators):
                return val1, val2
            elif any(indicator in val2_str for indicator in male_indicators):
                return val2, val1
            elif any(indicator in val1_str for indicator in female_indicators):
                return val2, val1
            elif any(indicator in val2_str for indicator in female_indicators):
                return val1, val2
        
        # Age patterns (older might be privileged in some contexts)
        if 'age' in col_lower:
            try:
                num1, num2 = float(val1), float(val2)
                return (val1, val2) if num1 > num2 else (val2, val1)
            except:
                pass
        
        # Default: use ordering (numeric or alphabetical)
        try:
            if all(isinstance(v, (int, float)) for v in values[:2]):
                return max(val1, val2), min(val1, val2)  # Higher numeric value is privileged
            else:
                sorted_vals = sorted(values[:2], key=str)
                return sorted_vals[1], sorted_vals[0]  # Later alphabetically is privileged
        except:
            return val1, val2

    def _determine_positive_values(self, values: List, column_name: str) -> Tuple:
        """Determine which value represents positive vs negative outcome using universal heuristics.""" 
        
        if len(values) < 2:
            return values[0] if values else 'Unknown', values[1] if len(values) > 1 else 'Unknown'
        
        val1, val2 = values[0], values[1]
        col_lower = column_name.lower()
        val1_str, val2_str = str(val1).lower(), str(val2).lower()
        
        # Outcome patterns - look for negative outcomes (what we want to predict/prevent)
        negative_outcome_indicators = [
            'dropout', 'drop', 'fail', 'failure', 'negative', 'bad', 'poor', 'low',
            'reject', 'denied', 'unsuccessful', 'terminate', 'quit', 'leave'
        ]
        
        positive_outcome_indicators = [
            'graduate', 'pass', 'success', 'positive', 'good', 'high', 'excellent',
            'accept', 'approved', 'successful', 'complete', 'finish', 'stay'
        ]
        
        # Check for explicit outcome indicators
        for indicator in negative_outcome_indicators:
            if indicator in val1_str:
                return val1, val2  # Negative outcome is "positive" class (what we predict)
            elif indicator in val2_str:
                return val2, val1
        
        for indicator in positive_outcome_indicators:
            if indicator in val1_str:
                return val2, val1  # Positive outcome is "negative" class (what we don't predict)
            elif indicator in val2_str:
                return val1, val2
        
        # Target/label column patterns
        if any(word in col_lower for word in ['target', 'label', 'outcome', 'result']):
            # Often 1 represents positive class, 0 negative class
            if val1_str in ['1', 'true', 'yes'] or val2_str in ['0', 'false', 'no']:
                return val1, val2
            elif val2_str in ['1', 'true', 'yes'] or val1_str in ['0', 'false', 'no']:
                return val2, val1
        
        # Default: first value alphabetically/numerically is positive class
        try:
            sorted_vals = sorted(values[:2], key=str)
            return sorted_vals[0], sorted_vals[1]
        except:
            return val1, val2

    def _get_diverse_examples(self, reference_data: pd.DataFrame, columns: Dict, 
                            sensitive_val, label_val, n_examples: int = 3) -> pd.DataFrame:
        """Get diverse examples with reduced size for single sample generation."""
        
        mask = ((reference_data[columns['sensitive_col']] == sensitive_val) & 
                (reference_data[columns['label_col']] == label_val))
        matching_data = reference_data[mask]
        
        if len(matching_data) == 0:
            diverse_pool = self._select_diverse_samples(reference_data, n_examples)
            return diverse_pool.sample(n=min(n_examples, len(diverse_pool)))
        
        if len(matching_data) <= n_examples:
            return matching_data
        
        # For single generation, use smaller example sets
        diverse_pool_size = min(len(matching_data), n_examples * 2)
        diverse_pool = self._select_diverse_samples(matching_data, diverse_pool_size)
        
        return diverse_pool.sample(n=n_examples)

    def _select_diverse_samples(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Select diverse samples using statistical diversity metrics."""
        
        if len(data) <= n_samples:
            return data
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            return data.sample(n=n_samples, random_state=42)
        
        try:
            try:
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                scaler = StandardScaler()
                numeric_data = data[numeric_cols].fillna(data[numeric_cols].mean())
                normalized_data = scaler.fit_transform(numeric_data)
                
                n_clusters = min(n_samples, len(data), 8)  # Reduced clusters for speed
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)  # Reduced iterations
                clusters = kmeans.fit_predict(normalized_data)
                
                diverse_samples = []
                for cluster_id in range(n_clusters):
                    cluster_mask = clusters == cluster_id
                    if cluster_mask.sum() > 0:
                        cluster_data = data[cluster_mask]
                        cluster_numeric = normalized_data[cluster_mask]
                        
                        cluster_center = kmeans.cluster_centers_[cluster_id]
                        distances = np.linalg.norm(cluster_numeric - cluster_center, axis=1)
                        closest_idx = np.argmin(distances)
                        diverse_samples.append(cluster_data.iloc[closest_idx])
                
                result = pd.DataFrame(diverse_samples).drop_duplicates()
                return result
                
            except ImportError:
                return self._quantile_based_sampling(data, numeric_cols, n_samples)
                
        except Exception:
            return data.sample(n=n_samples, random_state=42)

    def _quantile_based_sampling(self, data: pd.DataFrame, numeric_cols: List[str], n_samples: int) -> pd.DataFrame:
        """Fallback diversity sampling using quantiles."""
        
        if len(numeric_cols) == 0:
            return data.sample(n=n_samples, random_state=42)
        
        primary_col = numeric_cols[0]
        quantiles = np.linspace(0.1, 0.9, n_samples)
        
        diverse_samples = []
        used_indices = set()
        
        for q in quantiles:
            target_val = data[primary_col].quantile(q)
            distances = np.abs(data[primary_col] - target_val)
            sorted_indices = distances.sort_values().index
            
            for idx in sorted_indices:
                if idx not in used_indices:
                    diverse_samples.append(data.loc[idx])
                    used_indices.add(idx)
                    break
        
        result = pd.DataFrame(diverse_samples).drop_duplicates()
        return result

    def _create_provider_optimized_prompt(self, dataset_name: str, target_spec: Dict,
                                        reference_data: pd.DataFrame, schema: Dict,
                                        columns: Dict) -> str:
        """Create a prompt optimized for the specific provider."""
        
        # Get components
        sensitive_val, label_val = self._get_column_values_for_category(schema, columns, target_spec['category'], reference_data)
        diverse_examples = self._get_diverse_examples(reference_data, columns, sensitive_val, label_val, n_examples=2)
        essential_schema = self._create_essential_schema_description(schema, columns)
        
        # Create examples
        examples_text = ""
        for i, (_, row) in enumerate(diverse_examples.iterrows()):
            if i < 2:
                examples_text += f"Example {i+1}: {row.to_json()}\n"

        # Format logical relationships briefly
        logical_rules_text = ""
        logical_rels = schema.get('logical_relationships', {})
        rules = []
        
        for rel in logical_rels.get('numeric_constraints', []):
            if 'constraint' in rel:
                rules.append(f"- {rel['constraint']}")
        
        if rules and len(rules) <= 3:  # Only include if few rules
            logical_rules_text = f"\nIMPORTANT RULES:\n" + "\n".join(rules[:3])

        if self.provider == "huggingface":
            # Hugging Face models often work better with more conversational prompts
            prompt = (
                f"Task: Generate a synthetic data record for the {dataset_name} dataset.\n\n"
                f"Requirements:\n"
                f"- {columns['sensitive_col']} must be: {json.dumps(sensitive_val)}\n"
                f"- {columns['label_col']} must be: {json.dumps(label_val)}\n\n"
                f"Examples:\n{examples_text}\n"
                f"Schema (use similar field names and value types):\n{essential_schema}"
                f"{logical_rules_text}\n\n"
                f"Generate one record in valid JSON format (no extra text):\n"
            )
        else:
            # DeepSeek, OpenAI, and other OpenAI-compatible models
            prompt = (
                f"Generate 1 realistic synthetic record for the {dataset_name} dataset.\n\n"
                f"RESPONSE FORMAT: Return ONLY a valid JSON object. No explanations, no markdown.\n"
                f"Format: {{'field1': 'value1', 'field2': 'value2'}}\n\n"
                f"REQUIRED VALUES:\n"
                f"- {columns['sensitive_col']}: {json.dumps(sensitive_val)}\n"
                f"- {columns['label_col']}: {json.dumps(label_val)}\n\n"
                f"EXAMPLES:\n{examples_text}\n"
                f"SCHEMA (use similar field names and value types):\n{essential_schema}"
                f"{logical_rules_text}\n\n"
                f"Generate 1 record as JSON:"
            )
        
        return prompt

    def _create_essential_schema_description(self, schema: Dict, columns: Dict) -> str:
        """Create a simplified schema description with only essential information."""
        
        essential_schema = {}
        
        # Focus on key columns and a few representative ones
        important_cols = [columns['sensitive_col'], columns['label_col']]
        
        # Add a few more representative columns
        all_cols = list(schema['columns'].keys())
        for col in all_cols[:10]:  # Limit to first 10 columns
            if col not in important_cols:
                important_cols.append(col)
            if len(important_cols) >= 8:  # Cap at 8 total columns for prompt brevity
                break
        
        for col_name in important_cols:
            if col_name not in schema['columns']:
                continue
                
            col_schema = schema['columns'][col_name]
            type_category = col_schema.get('type_category', 'unknown')
            constraints = col_schema.get('constraints', {})
            
            if type_category == 'categorical':
                allowed_values = constraints.get('allowed_values', [])
                essential_schema[col_name] = {
                    'type': 'categorical',
                    'options': allowed_values[:5]  # Limit options
                }
            
            elif type_category == 'numeric':
                min_val = constraints.get('min_value')
                max_val = constraints.get('max_value')
                is_integer = constraints.get('is_integer', False)
                
                essential_schema[col_name] = {
                    'type': 'integer' if is_integer else 'float',
                    'range': f"[{min_val}, {max_val}]" if min_val is not None and max_val is not None else "numeric"
                }
            
            else:
                essential_schema[col_name] = {'type': type_category}
        
        return json.dumps(essential_schema, indent=1)

    def _make_api_call(self, prompt: str) -> Optional[str]:
        """Make API call based on provider type."""
        
        if self.api_type == "openai_compatible":
            return self._call_openai_compatible_api(prompt)
        elif self.api_type == "huggingface_inference":
            return self._call_huggingface_inference_api(prompt)
        elif self.api_type == "transformers_local":
            return self._call_local_transformers(prompt)
        else:
            raise ValueError(f"Unknown API type: {self.api_type}")

    def _call_openai_compatible_api(self, prompt: str) -> Optional[str]:
        """Call OpenAI-compatible API (DeepSeek, OpenAI, etc.)."""
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2048,
            "temperature": 0.8,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
            "seed": random.randint(1, 1000000)
        }

        try:
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
            
        except Exception as e:
            return None

    def _call_huggingface_inference_api(self, prompt: str) -> Optional[str]:
        """Call Hugging Face Inference API."""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.8,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }

        try:
            response = requests.post(f"{self.base_url}/{self.model_name}", 
                                json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            response_data = response.json()
            
            # DEBUG: Print what we're getting
            print(f"DEBUG: API Response: {response_data}")
            print(f"DEBUG: Response type: {type(response_data)}")
            
            # Handle different response formats
            if isinstance(response_data, list) and len(response_data) > 0:
                if 'generated_text' in response_data[0]:
                    return response_data[0]['generated_text']
                elif 'text' in response_data[0]:
                    return response_data[0]['text']
            elif isinstance(response_data, dict):
                if 'generated_text' in response_data:
                    return response_data['generated_text']
                elif 'text' in response_data:
                    return response_data['text']
            
            # Handle error responses
            if 'error' in str(response_data).lower():
                print(f"  ⚠️  Hugging Face API returned error: {response_data}")
                return None
            
            return str(response_data)
            
        except Exception as e:
            print(f"DEBUG: Exception in API call: {e}")
            return None
        
    def _call_local_transformers(self, prompt: str) -> Optional[str]:
        """Call local Hugging Face transformers model."""
        
        try:
            # Generate text using the pipeline
            result = self.generator(
                prompt,
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            if result and len(result) > 0:
                return result[0]['generated_text']
            
            return None
            
        except Exception as e:
            return None

    def _generate_one_sample_universal(self, dataset_name: str, target_spec: Dict, 
                                     reference_data: pd.DataFrame, schema: Dict, columns: Dict) -> Optional[Dict]:
        """Generate one sample using the configured provider."""
        
        # Create provider-optimized prompt
        prompt = self._create_provider_optimized_prompt(dataset_name, target_spec, reference_data, schema, columns)
        
        # Make API call
        raw_text = self._make_api_call(prompt)
        
        if not raw_text:
            return None
        
        # Enhanced JSON extraction
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
            r'\{[^{}]+\}',  # Simple objects
            r'\{[\s\S]*?\}'  # Most permissive
        ]
        
        for pattern in json_patterns:
            json_matches = re.findall(pattern, raw_text)
            
            for json_match in json_matches:
                try:
                    sample = json.loads(json_match)
                    
                    if isinstance(sample, dict) and len(sample) > 5:
                        if self._universal_sample_validation(sample, schema, columns, target_spec):
                            self._track_generated_sample(sample, target_spec['category'])
                            return sample
                            
                except json.JSONDecodeError:
                    cleaned_json = self._enhanced_json_cleaning(json_match)
                    if cleaned_json:
                        try:
                            sample = json.loads(cleaned_json)
                            if isinstance(sample, dict) and len(sample) > 5:
                                if self._universal_sample_validation(sample, schema, columns, target_spec):
                                    self._track_generated_sample(sample, target_spec['category'])
                                    return sample
                        except:
                            continue
        
        return None

    def _enhanced_json_cleaning(self, raw_json: str) -> Optional[str]:
        """Enhanced JSON cleaning with multiple strategies."""
        try:
            cleaned = raw_json.strip()
            
            # Remove markdown
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
            
            # Extract JSON object bounds
            json_start = cleaned.find('{')
            json_end = cleaned.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned = cleaned[json_start:json_end+1]
            
            # Fix trailing commas
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            
            # Fix unquoted keys
            cleaned = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
            
            # Fix single quotes
            cleaned = cleaned.replace("'", '"')
            
            # Fix unquoted string values (basic)
            cleaned = re.sub(r':\s*([^",\{\[\]\}\s][^",\}\]]*)\s*([,\}])', r': "\1"\2', cleaned)
            
            # Test parse
            json.loads(cleaned)
            return cleaned
            
        except:
            return None

    def _universal_sample_validation(self, sample: Dict, schema: Dict, columns: Dict, target_spec: Dict) -> bool:
        """Streamlined validation for faster processing."""
        
        # Quick essential checks
        if not isinstance(sample, dict) or len(sample) < 5:
            return False
        
        # Check required columns
        if columns['sensitive_col'] not in sample or columns['label_col'] not in sample:
            return False
        
        # Quick type validation for key columns
        validation_count = 0
        total_checks = 0
        
        for col_name, value in sample.items():
            if col_name in schema['columns']:
                col_schema = schema['columns'][col_name]
                if self._quick_field_validation(value, col_schema):
                    validation_count += 1
                total_checks += 1
                
                # Early exit if we've checked enough
                if total_checks >= 10:
                    break
        
        # Pass if at least 70% valid (more lenient for speed)
        success_rate = validation_count / total_checks if total_checks > 0 else 0
        return success_rate >= 0.7

    def _quick_field_validation(self, value: Any, col_schema: Dict) -> bool:
        """Quick field validation for speed."""
        
        type_category = col_schema.get('type_category', 'unknown')
        constraints = col_schema.get('constraints', {})
        
        if type_category == 'categorical':
            allowed_values = constraints.get('allowed_values', [])
            if not allowed_values:
                return True
            
            # Quick check
            return (value in allowed_values or 
                   str(value) in [str(v) for v in allowed_values[:10]])  # Limit check for speed
        
        elif type_category == 'numeric':
            try:
                num_val = float(value)
                min_val = constraints.get('min_value')
                max_val = constraints.get('max_value')
                
                # Allow generous bounds
                if min_val is not None and num_val < min_val * 0.8:
                    return False
                if max_val is not None and num_val > max_val * 1.2:
                    return False
                
                return True
                        
            except:
                return False
        
        else:
            return True  # Be lenient for other types

    def _track_generated_sample(self, sample: Dict, category: str) -> None:
        """Track generated samples."""
        
        if len(self._generated_samples_cache) > 500:  # Reduced cache size
            self._generated_samples_cache = self._generated_samples_cache[-250:]
        
        self._generated_samples_cache.append({
            'sample': sample,
            'category': category,
            'timestamp': time.time()
        })

    def generate_samples(self, dataset_name: str, target_specs: List[Dict], reference_data: pd.DataFrame) -> List[Dict]:
        """Main entry point - SINGLE SAMPLE MODE with multi-provider support."""
        
        # Extract schema
        schema = self._get_or_extract_schema(dataset_name, reference_data)
        
        # Identify key columns
        columns = self._identify_sensitive_and_label_columns(dataset_name, schema, reference_data)
        
        # Execute single-threaded generation
        return self._execute_single_generation(dataset_name, target_specs, reference_data, schema, columns)

    def _execute_single_generation(self, dataset_name: str, target_specs: List[Dict], 
                                  reference_data: pd.DataFrame, schema: Dict, columns: Dict) -> List[Dict]:
        """Execute single sample generation with multi-worker support."""
        
        # Create task list
        tasks_to_submit = []
        for spec in target_specs:
            for _ in range(spec['count']):
                tasks_to_submit.append(spec.copy())

        if not tasks_to_submit:
            return []

        print(f"🎯 Starting {self.provider.upper()} generation with {self.max_workers} workers for {len(tasks_to_submit)} samples...")
        
        all_samples = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._generate_one_sample_universal, dataset_name, task, 
                               reference_data, schema, columns): task 
                for task in tasks_to_submit
            }
            
            for future in tqdm(as_completed(futures), total=len(tasks_to_submit), desc=f"{self.provider.title()} Generation"):
                result = future.result()
                if result:
                    all_samples.append(result)
        
        print(f"\n✅ {self.provider.upper()} generation completed: {len(all_samples)}/{len(tasks_to_submit)} samples")
        
        # Report success rate by category
        category_counts = {}
        for task in tasks_to_submit:
            cat = task['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        category_successes = {}
        for sample_info in self._generated_samples_cache[-len(all_samples):]:
            cat = sample_info['category']
            category_successes[cat] = category_successes.get(cat, 0) + 1
        
        for category in category_counts:
            success_rate = (category_successes.get(category, 0) / category_counts[category]) * 100
            print(f"  📊 {category}: {category_successes.get(category, 0)}/{category_counts[category]} ({success_rate:.1f}% success)")
        
        return all_samples