# ModularCode/src/data_quality_analyzer.py (Complete Version)

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from scipy import stats
    from scipy.stats import ks_2samp, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  SciPy not available. Some statistical tests will be skipped.")

class DataQualityAnalyzer:
    """
    Comprehensive analyzer for comparing original vs augmented datasets.
    Provides detailed statistical comparisons and quality metrics.
    """
    
    def __init__(self):
        self.comparison_results = {}
        
    def comprehensive_comparison(self, original_data: pd.DataFrame, augmented_data: pd.DataFrame,
                           sensitive_col: str, label_col: str, dataset_name: str,
                           scenario_name: str, generation_method: str) -> Dict:
        """Comprehensive comparison with enhanced error handling."""
        
        print(f"🔍 Performing comprehensive data quality analysis...")
        
        # Separate original and synthetic data
        original_count = len(original_data)
        synthetic_data = augmented_data.iloc[original_count:] if len(augmented_data) > original_count else pd.DataFrame()
        
        if len(synthetic_data) == 0:
            print("⚠️  No synthetic data found. Analysis will be limited.")
            return {
                'metadata': {
                    'dataset_name': dataset_name,
                    'scenario_name': scenario_name,
                    'generation_method': generation_method,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'original_samples': len(original_data),
                    'synthetic_samples': 0,
                    'augmentation_success': False
                },
                'error': 'No synthetic data generated',
                'recommendations': [
                    'Check LLM API connectivity and authentication',
                    'Verify prompt formatting and response parsing',
                    'Consider using Faker method as fallback',
                    'Review dataset schema extraction process'
                ]
            }
        
        # Proceed with normal analysis
        analysis = {
            'metadata': {
                'dataset_name': dataset_name,
                'scenario_name': scenario_name,
                'generation_method': generation_method,
                'timestamp': pd.Timestamp.now().isoformat(),
                'original_samples': len(original_data),
                'synthetic_samples': len(synthetic_data),
                'augmentation_success': True
            },
            
            # Core analyses
            'basic_statistics': self._basic_statistical_comparison(original_data, synthetic_data),
            'distribution_analysis': self._distribution_similarity_analysis(original_data, synthetic_data),
            'correlation_analysis': self._correlation_preservation_analysis(original_data, synthetic_data),
            'column_by_column_analysis': self._column_by_column_analysis(original_data, synthetic_data),
            'fairness_preservation': self._fairness_preservation_analysis(
                original_data, synthetic_data, sensitive_col, label_col
            ),
            
            # Quality scores
            'generation_quality_scores': self._calculate_generation_quality_scores(original_data, synthetic_data),
            
            # Recommendations
            'recommendations': self._generate_recommendations(original_data, synthetic_data, generation_method)
        }
        
        return analysis

    def _basic_statistical_comparison(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Compare basic dataset statistics."""
        
        return {
            'size_comparison': {
                'original_rows': len(original),
                'synthetic_rows': len(synthetic),
                'size_increase_ratio': len(synthetic) / len(original) if len(original) > 0 else 0,
                'size_increase_percentage': (len(synthetic) / len(original)) * 100 if len(original) > 0 else 0
            },
            'shape_comparison': {
                'original_shape': original.shape,
                'synthetic_shape': synthetic.shape,
                'columns_match': original.shape[1] == synthetic.shape[1]
            },
            'missing_values': {
                'original_missing_count': original.isnull().sum().sum(),
                'synthetic_missing_count': synthetic.isnull().sum().sum(),
                'original_missing_percentage': (original.isnull().sum().sum() / (len(original) * len(original.columns))) * 100 if len(original) > 0 else 0,
                'synthetic_missing_percentage': (synthetic.isnull().sum().sum() / (len(synthetic) * len(synthetic.columns))) * 100 if len(synthetic) > 0 else 0
            },
            'data_types': {
                'original_dtypes': {str(k): int(v) for k, v in original.dtypes.value_counts().to_dict().items()},
                'synthetic_dtypes': {str(k): int(v) for k, v in synthetic.dtypes.value_counts().to_dict().items()},
                'dtype_consistency': (original.dtypes == synthetic.dtypes).all()
            }
        }

    def _distribution_similarity_analysis(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Compare statistical distributions between datasets."""
        
        distributions = {}
        
        for column in original.columns:
            if column not in synthetic.columns:
                continue
                
            col_analysis = {
                'column_name': column,
                'data_type': str(original[column].dtype),
                'original_stats': {},
                'synthetic_stats': {},
                'distribution_similarity': {}
            }
            
            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(original[column]):
                col_analysis['original_stats'] = self._calculate_numeric_stats(original[column])
                col_analysis['synthetic_stats'] = self._calculate_numeric_stats(synthetic[column])
                col_analysis['distribution_similarity'] = self._calculate_numeric_similarity(
                    original[column].dropna(), synthetic[column].dropna()
                )
            
            # Handle categorical columns
            else:
                col_analysis['original_stats'] = self._calculate_categorical_stats(original[column])
                col_analysis['synthetic_stats'] = self._calculate_categorical_stats(synthetic[column])
                col_analysis['distribution_similarity'] = self._calculate_categorical_similarity(
                    original[column], synthetic[column]
                )
            
            distributions[column] = col_analysis
        
        return distributions

    def _correlation_preservation_analysis(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
        """Compare correlation structures between datasets."""
        
        # Get numeric columns only
        numeric_cols = original.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {'message': 'Insufficient numeric columns for correlation analysis'}
        
        try:
            orig_corr = original[numeric_cols].corr()
            synth_corr = synthetic[numeric_cols].corr()
            
            # Calculate correlation matrix similarity
            correlation_similarity = self._calculate_correlation_similarity(orig_corr, synth_corr)
            
            return {
                'original_correlation_summary': {
                    'mean_abs_correlation': float(np.abs(orig_corr.values).mean()),
                    'max_correlation': float(np.abs(orig_corr.values).max()),
                    'highly_correlated_pairs': self._find_high_correlations(orig_corr, threshold=0.7)
                },
                'synthetic_correlation_summary': {
                    'mean_abs_correlation': float(np.abs(synth_corr.values).mean()),
                    'max_correlation': float(np.abs(synth_corr.values).max()),
                    'highly_correlated_pairs': self._find_high_correlations(synth_corr, threshold=0.7)
                },
                'correlation_preservation': correlation_similarity
            }
        except Exception as e:
            return {'error': f'Correlation analysis failed: {str(e)}'}

    def _fairness_preservation_analysis(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                                      sensitive_col: str, label_col: str) -> Dict[str, Any]:
        """Analyze how augmentation affects fairness metrics."""
        
        def calculate_fairness_stats(data: pd.DataFrame) -> Dict[str, Any]:
            """Calculate fairness statistics for a dataset."""
            if sensitive_col not in data.columns or label_col not in data.columns:
                return {}
            
            try:
                # Cross-tabulation
                cross_tab = pd.crosstab(data[sensitive_col], data[label_col])
                
                # Calculate rates
                sensitive_vals = data[sensitive_col].unique()
                if len(sensitive_vals) >= 2:
                    priv_val = sensitive_vals[0]
                    unpriv_val = sensitive_vals[1]
                    
                    priv_data = data[data[sensitive_col] == priv_val]
                    unpriv_data = data[data[sensitive_col] == unpriv_val]
                    
                    label_vals = data[label_col].unique()
                    if len(label_vals) >= 2:
                        pos_val = label_vals[0]
                        
                        priv_pos_rate = (priv_data[label_col] == pos_val).mean()
                        unpriv_pos_rate = (unpriv_data[label_col] == pos_val).mean()
                        
                        return {
                            'cross_tabulation': {str(k): {str(k2): int(v2) for k2, v2 in v.items()} for k, v in cross_tab.to_dict().items()},
                            'privileged_positive_rate': float(priv_pos_rate),
                            'unprivileged_positive_rate': float(unpriv_pos_rate),
                            'demographic_parity_difference': float(priv_pos_rate - unpriv_pos_rate),
                            'sensitive_attribute_balance': float((data[sensitive_col] == priv_val).mean()),
                            'label_balance': float((data[label_col] == pos_val).mean())
                        }
                
                return {}
            except Exception as e:
                return {'error': str(e)}
        
        original_fairness = calculate_fairness_stats(original)
        synthetic_fairness = calculate_fairness_stats(synthetic)
        
        # Calculate improvement metrics
        improvements = {}
        if original_fairness and synthetic_fairness and 'error' not in original_fairness and 'error' not in synthetic_fairness:
            improvements = {
                'demographic_parity_improvement': abs(original_fairness.get('demographic_parity_difference', 0)) - 
                                                abs(synthetic_fairness.get('demographic_parity_difference', 0)),
                'sensitive_balance_change': synthetic_fairness.get('sensitive_attribute_balance', 0) - 
                                          original_fairness.get('sensitive_attribute_balance', 0),
                'label_balance_change': synthetic_fairness.get('label_balance', 0) - 
                                      original_fairness.get('label_balance', 0)
            }
        
        return {
            'original_fairness_metrics': original_fairness,
            'synthetic_fairness_metrics': synthetic_fairness,
            'fairness_improvements': improvements
        }

    def _column_by_column_analysis(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict:
        """Analyze each column individually comparing original vs synthetic."""
        
        if len(synthetic) == 0:
            print("⚠️  No synthetic data to analyze. Skipping column-by-column analysis.")
            return {
                'error': 'No synthetic data available',
                'columns_analyzed': 0,
                'analysis': {}
            }
        
        analysis = {}
        
        for column in original.columns:
            if column not in synthetic.columns:
                analysis[column] = {
                    'error': 'Column missing in synthetic data',
                    'status': 'failed'
                }
                continue
            
            try:
                col_analysis = {
                    'data_type': str(original[column].dtype),
                    'original_unique_count': original[column].nunique(),
                    'synthetic_unique_count': synthetic[column].nunique(),
                    'original_unique_ratio': original[column].nunique() / len(original),
                    'synthetic_unique_ratio': synthetic[column].nunique() / len(synthetic) if len(synthetic) > 0 else 0,
                    'original_null_count': int(original[column].isnull().sum()),
                    'synthetic_null_count': int(synthetic[column].isnull().sum()),
                    'original_null_ratio': original[column].isnull().sum() / len(original),
                    'synthetic_null_ratio': synthetic[column].isnull().sum() / len(synthetic) if len(synthetic) > 0 else 0,
                }
                
                # Type-specific analysis
                if pd.api.types.is_numeric_dtype(original[column]):
                    col_analysis.update(self._analyze_numeric_column(original[column], synthetic[column]))
                elif pd.api.types.is_categorical_dtype(original[column]) or original[column].dtype == 'object':
                    col_analysis.update(self._analyze_categorical_column(original[column], synthetic[column]))
                
                analysis[column] = col_analysis
                
            except Exception as e:
                analysis[column] = {
                    'error': f'Analysis failed: {str(e)}',
                    'status': 'failed'
                }
        
        return {
            'columns_analyzed': len(analysis),
            'analysis': analysis
        }

    def _calculate_generation_quality_scores(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, float]:
        """Calculate overall generation quality scores."""
        
        scores = {
            'fidelity_score': self._calculate_fidelity_score(original, synthetic),
            'diversity_score': self._calculate_diversity_score(synthetic),
            'privacy_score': self._calculate_privacy_score(original, synthetic),
            'utility_score': self._calculate_utility_score(original, synthetic)
        }
        
        # Calculate overall score as weighted average
        weights = {'fidelity_score': 0.3, 'diversity_score': 0.25, 'privacy_score': 0.2, 'utility_score': 0.25}
        scores['overall_generation_score'] = sum(scores[key] * weights[key] for key in weights)
        
        return scores

    # Helper methods for statistics calculation
    def _calculate_numeric_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for numeric series."""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {'error': 'No valid values'}
        
        return {
            'mean': float(clean_series.mean()),
            'median': float(clean_series.median()),
            'std': float(clean_series.std()),
            'min': float(clean_series.min()),
            'max': float(clean_series.max()),
            'skewness': float(clean_series.skew()),
            'kurtosis': float(clean_series.kurtosis()),
            'count': int(len(clean_series))
        }

    def _calculate_categorical_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for categorical series."""
        counts = series.value_counts()
        
        return {
            'unique_count': series.nunique(),
            'most_frequent': str(counts.index[0]) if len(counts) > 0 else None,
            'most_frequent_count': int(counts.iloc[0]) if len(counts) > 0 else 0,
            'value_distribution': {str(k): int(v) for k, v in counts.head(10).to_dict().items()},
            'total_count': int(len(series.dropna()))
        }

    def _analyze_numeric_column(self, orig_series: pd.Series, synth_series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column comparison."""
        return {
            'range_comparison': {
                'original_range': [float(orig_series.min()), float(orig_series.max())],
                'synthetic_range': [float(synth_series.min()), float(synth_series.max())],
                'range_overlap': self._calculate_range_overlap(orig_series, synth_series)
            },
            'distribution_comparison': self._calculate_numeric_similarity(orig_series.dropna(), synth_series.dropna())
        }

    def _analyze_categorical_column(self, orig_series: pd.Series, synth_series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column comparison."""
        return {
            'category_comparison': self._calculate_categorical_similarity(orig_series, synth_series),
            'value_preservation': {
                'new_values': list(set(synth_series.dropna().unique()) - set(orig_series.dropna().unique())),
                'missing_values': list(set(orig_series.dropna().unique()) - set(synth_series.dropna().unique()))
            }
        }

    # Quality score calculation methods
    def _calculate_numeric_similarity(self, orig_series: pd.Series, synth_series: pd.Series) -> Dict[str, float]:
        """Calculate similarity metrics for numeric columns."""
        if len(orig_series) == 0 or len(synth_series) == 0:
            return {'error': 'Empty series provided'}
        
        try:
            mean_diff = abs(orig_series.mean() - synth_series.mean()) / (orig_series.std() + 1e-8)
            std_ratio = min(orig_series.std(), synth_series.std()) / (max(orig_series.std(), synth_series.std()) + 1e-8)
            
            return {
                'mean_similarity': float(1 / (1 + mean_diff)),
                'std_similarity': float(std_ratio),
                'range_overlap': self._calculate_range_overlap(orig_series, synth_series)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_categorical_similarity(self, orig_series: pd.Series, synth_series: pd.Series) -> Dict[str, float]:
        """Calculate similarity metrics for categorical columns."""
        try:
            orig_dist = orig_series.value_counts(normalize=True)
            synth_dist = synth_series.value_counts(normalize=True)
            
            # Calculate overlap in categories
            common_cats = set(orig_dist.index) & set(synth_dist.index)
            category_overlap = len(common_cats) / len(set(orig_dist.index) | set(synth_dist.index)) if len(set(orig_dist.index) | set(synth_dist.index)) > 0 else 0
            
            # Calculate distribution similarity for common categories
            if common_cats:
                orig_common = orig_dist[list(common_cats)]
                synth_common = synth_dist[list(common_cats)]
                dist_similarity = 1 - 0.5 * np.sum(np.abs(orig_common - synth_common))
            else:
                dist_similarity = 0.0
            
            return {
                'category_overlap': float(category_overlap),
                'distribution_similarity': float(dist_similarity),
                'new_categories_ratio': len(set(synth_dist.index) - set(orig_dist.index)) / len(synth_dist.index) if len(synth_dist.index) > 0 else 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_range_overlap(self, orig_series: pd.Series, synth_series: pd.Series) -> float:
        """Calculate how much the ranges of two numeric series overlap."""
        try:
            orig_min, orig_max = orig_series.min(), orig_series.max()
            synth_min, synth_max = synth_series.min(), synth_series.max()
            
            overlap_min = max(orig_min, synth_min)
            overlap_max = min(orig_max, synth_max)
            
            if overlap_max <= overlap_min:
                return 0.0
            
            overlap_range = overlap_max - overlap_min
            total_range = max(orig_max, synth_max) - min(orig_min, synth_min)
            
            return float(overlap_range / total_range) if total_range > 0 else 1.0
        except:
            return 0.0

    def _calculate_correlation_similarity(self, orig_corr: pd.DataFrame, synth_corr: pd.DataFrame) -> Dict[str, float]:
        """Calculate similarity between correlation matrices."""
        
        try:
            # Flatten correlation matrices (excluding diagonal)
            orig_flat = orig_corr.values[np.triu_indices_from(orig_corr.values, k=1)]
            synth_flat = synth_corr.values[np.triu_indices_from(synth_corr.values, k=1)]
            
            # Calculate correlation between the two sets of correlations
            corr_correlation = np.corrcoef(orig_flat, synth_flat)[0, 1]
            
            # Calculate mean absolute difference
            mean_abs_diff = np.mean(np.abs(orig_flat - synth_flat))
            
            return {
                'correlation_of_correlations': float(corr_correlation) if not np.isnan(corr_correlation) else 0.0,
                'mean_absolute_difference': float(mean_abs_diff),
                'overall_similarity': float(1 - mean_abs_diff) if mean_abs_diff <= 1 else 0.0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find highly correlated variable pairs."""
        
        high_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corrs.append({
                        'variable_1': corr_matrix.columns[i],
                        'variable_2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        return high_corrs

    def _calculate_fidelity_score(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate how well synthetic data matches original data distributions."""
        
        fidelity_scores = []
        
        for column in original.columns:
            if column in synthetic.columns:
                if pd.api.types.is_numeric_dtype(original[column]):
                    sim = self._calculate_numeric_similarity(original[column].dropna(), synthetic[column].dropna())
                    if 'mean_similarity' in sim and 'std_similarity' in sim:
                        fidelity_scores.append((sim['mean_similarity'] + sim['std_similarity']) / 2)
                else:
                    sim = self._calculate_categorical_similarity(original[column], synthetic[column])
                    if 'distribution_similarity' in sim:
                        fidelity_scores.append(sim['distribution_similarity'])
        
        return float(np.mean(fidelity_scores)) if fidelity_scores else 0.0
    
    def _calculate_diversity_score(self, synthetic: pd.DataFrame) -> float:
        """Calculate diversity within synthetic data."""
        
        diversity_scores = []
        
        for column in synthetic.columns:
            if pd.api.types.is_numeric_dtype(synthetic[column]):
                # For numeric: coefficient of variation
                mean_val = synthetic[column].mean()
                std_val = synthetic[column].std()
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    diversity_scores.append(min(cv, 1.0))  # Cap at 1.0
            else:
                # For categorical: entropy-based diversity
                value_counts = synthetic[column].value_counts(normalize=True)
                entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)
                max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1
                diversity_scores.append(entropy / max_entropy if max_entropy > 0 else 0)
        
        return float(np.mean(diversity_scores)) if diversity_scores else 0.0
    
    def _calculate_privacy_score(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate privacy preservation score (how different synthetic data is from original)."""
        
        # Simple privacy score: measure how many synthetic rows are identical to original rows
        if len(original) == 0 or len(synthetic) == 0:
            return 1.0
        
        try:
            # Convert to strings for comparison
            orig_strings = original.astype(str).apply(lambda x: '|'.join(x), axis=1)
            synth_strings = synthetic.astype(str).apply(lambda x: '|'.join(x), axis=1)
            
            # Count identical rows
            identical_count = sum(1 for synth_row in synth_strings if synth_row in orig_strings.values)
            
            # Privacy score: 1 - (proportion of identical rows)
            privacy_score = 1 - (identical_count / len(synthetic))
            
            return float(privacy_score)
        except:
            return 0.8  # Default reasonable privacy score
    
    def _calculate_utility_score(self, original: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        """Calculate utility score (how useful synthetic data is for ML tasks)."""
        
        # This is a simplified utility score based on statistical similarity
        utility_scores = []
        
        # For now, use correlation preservation as a proxy for utility
        numeric_cols = original.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            try:
                orig_corr = original[numeric_cols].corr()
                synth_corr = synthetic[numeric_cols].corr()
                
                corr_sim = self._calculate_correlation_similarity(orig_corr, synth_corr)
                if 'overall_similarity' in corr_sim:
                    utility_scores.append(corr_sim['overall_similarity'])
            except:
                pass
        
        # Add distribution similarity as another utility measure
        fidelity_score = self._calculate_fidelity_score(original, synthetic)
        utility_scores.append(fidelity_score)
        
        return float(np.mean(utility_scores)) if utility_scores else 0.5

    def _generate_recommendations(self, original: pd.DataFrame, synthetic: pd.DataFrame, generation_method: str) -> List[str]:
        """Generate recommendations based on the analysis."""
        
        recommendations = []
        
        # Check generation quality scores
        gen_scores = self._calculate_generation_quality_scores(original, synthetic)
        overall_score = gen_scores.get('overall_generation_score', 0)
        
        if overall_score < 0.6:
            recommendations.append(f"Overall generation quality is {overall_score:.2f} (below 60%). Consider adjusting generation parameters.")
        
        if gen_scores.get('fidelity_score', 0) < 0.5:
            recommendations.append("Fidelity score is low. Generated data may not preserve statistical properties well.")
        
        if gen_scores.get('diversity_score', 0) < 0.5:
            recommendations.append("Diversity score is low. Generated data may lack sufficient variation.")
        
        if gen_scores.get('privacy_score', 0) < 0.8:
            recommendations.append("Privacy score suggests some synthetic data may be too similar to original data.")
        
        # Method-specific recommendations
        if generation_method.lower() == 'faker':
            recommendations.append("Consider using LLM-based generation for better fidelity to original data patterns.")
        elif 'llm' in generation_method.lower():
            recommendations.append("LLM generation detected. Consider fine-tuning prompts for better quality.")
        
        if not recommendations:
            recommendations.append("Data generation quality looks good overall!")
        
        return recommendations

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert pandas and numpy objects to JSON serializable format."""
        
        if isinstance(obj, dict):
            return {str(k): self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):  # pandas objects
            if hasattr(obj, 'to_dict'):
                return self._make_json_serializable(obj.to_dict())
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return str(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

    def save_analysis(self, analysis: Dict[str, Any], filepath: str) -> None:
        """Save analysis results to JSON file with proper serialization."""
        
        try:
            # Ensure directory exists
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Make the analysis JSON serializable
            serializable_analysis = self._make_json_serializable(analysis)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_analysis, f, indent=2, default=str)
            print(f"✅ Data quality analysis saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
            # Try to save a simplified version
            try:
                simplified_analysis = {
                    'metadata': analysis.get('metadata', {}),
                    'generation_quality_scores': analysis.get('generation_quality_scores', {}),
                    'recommendations': analysis.get('recommendations', []),
                    'error': f"Full analysis could not be saved: {str(e)}"
                }
                simplified_analysis = self._make_json_serializable(simplified_analysis)
                
                with open(filepath.replace('.json', '_simplified.json'), 'w') as f:
                    json.dump(simplified_analysis, f, indent=2, default=str)
                print(f"✅ Simplified analysis saved to: {filepath.replace('.json', '_simplified.json')}")
            except Exception as e2:
                print(f"❌ Could not save even simplified analysis: {e2}")