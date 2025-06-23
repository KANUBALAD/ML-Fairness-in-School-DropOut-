import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    # Updated imports for SDV 1.23.0+
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.sampling import Condition
    SDV_AVAILABLE = True
    print("✓ SDV library loaded successfully with conditional sampling support")
except ImportError as e:
    try:
        # Fallback for older SDV versions
        from sdv.tabular import CTGAN as CTGANSynthesizer, TVAE as TVAESynthesizer
        SDV_AVAILABLE = True
        print("⚠ Using legacy SDV API - conditional sampling may be limited")
    except ImportError:
        print("SDV library not found. Please install with: pip install sdv")
        SDV_AVAILABLE = False

class CTGANSyntheticGenerator:
    """
    Optimized CTGAN-based generator with proper conditional sampling API.
    """
    
    def __init__(self, epochs: int = 100, batch_size: int = 500, verbose: bool = True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.metadata = None
        
        if not SDV_AVAILABLE:
            raise ImportError("SDV library is required. Install with: pip install sdv")
    
    def fit_and_generate(self, original_data: pd.DataFrame, target_specs: List[Dict], 
                        dataset_name: str) -> pd.DataFrame:
        """
        Train CTGAN on original data and generate samples with proper conditional constraints.
        """
        
        print(f"Training CTGAN on {len(original_data)} samples for {self.epochs} epochs...")
        
        # Prepare data for CTGAN
        prepared_data = self._prepare_data_for_ctgan(original_data, dataset_name)
        
        # Create metadata for the dataset
        self.metadata = self._create_metadata(prepared_data, dataset_name)
        
        # Initialize and train CTGAN with updated API
        print("Initializing CTGAN with metadata...")
        self.model = CTGANSynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        
        print("Starting CTGAN training...")
        self.model.fit(prepared_data)
        print("✓ CTGAN training completed")
        
        # Generate samples for each category with proper conditional sampling
        all_synthetic = []
        
        for spec in target_specs:
            if spec['count'] > 0:
                print(f"Generating {spec['count']} samples for {spec['category']}")
                
                # Use corrected conditional sampling
                synthetic_batch = self._generate_ctgan_samples_with_conditions(spec, dataset_name)
                
                if len(synthetic_batch) > 0:
                    all_synthetic.append(synthetic_batch)
                    print(f"✓ Generated {len(synthetic_batch)} samples for {spec['category']}")
                else:
                    print(f"⚠ Failed to generate samples for {spec['category']}")
        
        if all_synthetic:
            synthetic_df = pd.concat(all_synthetic, ignore_index=True)
            
            # Ensure column compatibility
            synthetic_df = self._ensure_column_compatibility(synthetic_df, original_data)
            
            # Combine with original data
            augmented_data = pd.concat([original_data, synthetic_df], ignore_index=True)
            print(f"✓ Created augmented dataset with {len(augmented_data)} total samples")
            return augmented_data
        else:
            print("⚠ No synthetic samples generated, returning original data")
            return original_data
    
    def _generate_ctgan_samples_with_conditions(self, spec: Dict, dataset_name: str) -> pd.DataFrame:
        """Generate CTGAN samples using proper conditional sampling API."""
        
        samples_needed = spec['count']
        
        # Strategy 1: Try proper conditional sampling with Condition objects
        try:
            print(f"  Attempting conditional sampling using SDV Condition API...")
            
            # Create proper Condition object
            condition_dict = self._create_condition_dict(spec, dataset_name)
            print(f"    Condition: {condition_dict}")
            
            # Method 1: Using sample_conditions with Condition objects
            try:
                conditions = [Condition(condition_dict, num_rows=samples_needed)]
                synthetic_batch = self.model.sample_conditions(conditions=conditions)
                
                # Validate results
                if self._validate_conditional_results(synthetic_batch, spec, dataset_name):
                    print(f"  ✓ Conditional sampling successful via sample_conditions")
                    return self._post_process_samples(synthetic_batch, spec, dataset_name)
                else:
                    print(f"  ⚠ Conditional sampling produced incorrect distributions")
                    
            except Exception as e:
                print(f"  ✗ sample_conditions failed: {e}")
            
            # Method 2: Using sample_remaining_columns (for newer SDV versions)
            try:
                known_columns = pd.DataFrame([condition_dict] * samples_needed)
                synthetic_batch = self.model.sample_remaining_columns(known_columns)
                
                if self._validate_conditional_results(synthetic_batch, spec, dataset_name):
                    print(f"  ✓ Conditional sampling successful via sample_remaining_columns")
                    return self._post_process_samples(synthetic_batch, spec, dataset_name)
                    
            except Exception as e:
                print(f"  ✗ sample_remaining_columns failed: {e}")
            
            # Method 3: Using sample with conditions parameter (alternative API)
            try:
                synthetic_batch = self.model.sample(
                    num_rows=samples_needed,
                    conditions=condition_dict
                )
                
                if self._validate_conditional_results(synthetic_batch, spec, dataset_name):
                    print(f"  ✓ Conditional sampling successful via sample(conditions=...)")
                    return self._post_process_samples(synthetic_batch, spec, dataset_name)
                    
            except Exception as e:
                print(f"  ✗ sample(conditions=...) failed: {e}")
                
        except Exception as e:
            print(f"  ✗ Conditional setup failed: {e}")
        
        # Strategy 2: Rejection sampling with intelligent batching
        print(f"  Falling back to optimized rejection sampling...")
        try:
            return self._rejection_sampling_optimized(spec, dataset_name, samples_needed)
        except Exception as e:
            print(f"  ✗ Rejection sampling failed: {e}")
        
        # Strategy 3: Unconditional generation with forced values (last resort)
        print(f"  Last resort: Unconditional generation with forced values...")
        try:
            synthetic_batch = self.model.sample(num_rows=samples_needed)
            return self._post_process_samples(synthetic_batch, spec, dataset_name)
        except Exception as e:
            print(f"  ✗ All generation methods failed: {e}")
            return pd.DataFrame()
    
    def _create_condition_dict(self, spec: Dict, dataset_name: str) -> Dict:
        """Create proper condition dictionary for SDV Condition objects."""
        
        if dataset_name == 'brazil':
            return {
                'Gender': int(spec['sensitive_val']),  # Ensure integer type
                'Target': str(spec['label_val'])       # Ensure string type
            }
        elif dataset_name == 'africa':
            return {
                'gender': str(spec['sensitive_val']),
                'dropout': str(spec['label_val'])
            }
        elif dataset_name == 'india':
            return {
                'STUDENTGENDER': str(spec['sensitive_val']),
                'STUDENT_DROPOUT_STATUS': str(spec['label_val'])
            }
        
        return {}
    
    def _validate_conditional_results(self, samples: pd.DataFrame, spec: Dict, dataset_name: str) -> bool:
        """Validate that conditional sampling produced correct distributions."""
        
        if len(samples) == 0:
            return False
        
        try:
            if dataset_name == 'brazil':
                gender_col = 'Gender'
                target_col = 'Target'
                expected_gender = int(spec['sensitive_val'])
                expected_target = str(spec['label_val'])
            elif dataset_name == 'africa':
                gender_col = 'gender'
                target_col = 'dropout'
                expected_gender = str(spec['sensitive_val'])
                expected_target = str(spec['label_val'])
            elif dataset_name == 'india':
                gender_col = 'STUDENTGENDER'
                target_col = 'STUDENT_DROPOUT_STATUS'
                expected_gender = str(spec['sensitive_val'])
                expected_target = str(spec['label_val'])
            else:
                return False
            
            # Check if columns exist
            if gender_col not in samples.columns or target_col not in samples.columns:
                print(f"    Missing required columns: {gender_col}, {target_col}")
                return False
            
            # Calculate match percentages
            if dataset_name == 'brazil':
                gender_match = (samples[gender_col].astype(int) == expected_gender).mean()
                target_match = (samples[target_col].astype(str) == expected_target).mean()
            else:
                gender_match = (samples[gender_col].astype(str) == expected_gender).mean()
                target_match = (samples[target_col].astype(str) == expected_target).mean()
            
            print(f"    Validation: Gender match {gender_match:.1%}, Target match {target_match:.1%}")
            
            # Require at least 95% match for conditional validation
            return gender_match >= 0.95 and target_match >= 0.95
            
        except Exception as e:
            print(f"    Validation error: {e}")
            return False
    
    def _rejection_sampling_optimized(self, spec: Dict, dataset_name: str, samples_needed: int) -> pd.DataFrame:
        """Optimized rejection sampling with adaptive batch sizing."""
        
        all_valid_samples = []
        attempt = 0
        max_attempts = 8
        
        # Adaptive batch sizing based on estimated success rate
        estimated_success_rate = 0.15  # Start with 15% estimate
        
        while len(all_valid_samples) < samples_needed and attempt < max_attempts:
            attempt += 1
            
            # Calculate batch size based on need and estimated success rate
            remaining_needed = samples_needed - len(all_valid_samples)
            batch_size = min(
                int(remaining_needed / max(estimated_success_rate, 0.05)),  # Don't go below 5%
                2000  # Cap at 2000
            )
            
            try:
                # Generate batch
                synthetic_batch = self.model.sample(num_rows=batch_size)
                
                # Filter for target conditions
                valid_samples = self._filter_samples_strict(synthetic_batch, spec, dataset_name)
                
                if len(valid_samples) > 0:
                    all_valid_samples.extend(valid_samples.to_dict('records'))
                    
                    # Update success rate estimate
                    actual_success_rate = len(valid_samples) / batch_size
                    estimated_success_rate = 0.7 * estimated_success_rate + 0.3 * actual_success_rate
                    
                    print(f"    Rejection attempt {attempt}: {len(valid_samples)}/{batch_size} valid ({actual_success_rate:.1%} success, total: {len(all_valid_samples)})")
                else:
                    # Reduce success rate estimate if no samples found
                    estimated_success_rate *= 0.8
                    print(f"    Rejection attempt {attempt}: 0/{batch_size} valid samples found")
                
            except Exception as e:
                print(f"    Rejection attempt {attempt} failed: {e}")
        
        if len(all_valid_samples) >= samples_needed:
            final_samples = all_valid_samples[:samples_needed]
            result_df = pd.DataFrame(final_samples)
            print(f"  ✓ Rejection sampling successful: {len(final_samples)} samples")
            return self._post_process_samples(result_df, spec, dataset_name)
        
        elif len(all_valid_samples) > 0:
            # Use partial results and pad with forced values
            result_df = pd.DataFrame(all_valid_samples)
            remaining = samples_needed - len(result_df)
            
            if remaining > 0:
                try:
                    padding_batch = self.model.sample(num_rows=remaining)
                    padding_batch = self._post_process_samples(padding_batch, spec, dataset_name)
                    result_df = pd.concat([result_df, padding_batch], ignore_index=True)
                except:
                    pass
            
            print(f"  ⚠ Partial rejection sampling: {len(result_df)} samples")
            return self._post_process_samples(result_df, spec, dataset_name)
        
        else:
            print(f"  ✗ Rejection sampling failed completely")
            return pd.DataFrame()
    
    def _filter_samples_strict(self, samples: pd.DataFrame, spec: Dict, dataset_name: str) -> pd.DataFrame:
        """Strict filtering for rejection sampling."""
        
        try:
            if dataset_name == 'brazil':
                mask = (samples['Gender'].astype(int) == int(spec['sensitive_val'])) & \
                       (samples['Target'].astype(str) == str(spec['label_val']))
            elif dataset_name == 'africa':
                mask = (samples['gender'].astype(str) == str(spec['sensitive_val'])) & \
                       (samples['dropout'].astype(str) == str(spec['label_val']))
            elif dataset_name == 'india':
                mask = (samples['STUDENTGENDER'].astype(str) == str(spec['sensitive_val'])) & \
                       (samples['STUDENT_DROPOUT_STATUS'].astype(str) == str(spec['label_val']))
            else:
                return pd.DataFrame()
            
            return samples[mask]
            
        except Exception as e:
            print(f"      Filtering error: {e}")
            return pd.DataFrame()
    
    def _create_metadata(self, data: pd.DataFrame, dataset_name: str) -> SingleTableMetadata:
        """Create optimized metadata for conditional generation."""
        
        print("Creating metadata for conditional generation...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        
        # Set specific column types for better conditional generation
        if dataset_name == 'brazil':
            # Primary conditional columns - these are critical for conditioning
            primary_categorical = ['Gender', 'Target']
            
            # Secondary categorical columns
            secondary_categorical = [
                'Tuition fees up to date', 'Displaced', 'Educational special needs', 
                'Debtor', 'Scholarship holder', 'International', 'Daytime/evening attendance',
                'Marital status', 'Application mode', 'Course', 'Previous qualification',
                'Nacionality', 'Mother\'s qualification', 'Father\'s qualification',
                'Mother\'s occupation', 'Father\'s occupation'
            ]
            
            # Set primary columns with explicit categorical type
            for col in primary_categorical:
                if col in data.columns:
                    metadata.update_column(column_name=col, sdtype='categorical')
                    print(f"  Set {col} as primary categorical (conditional target)")
            
            # Set secondary categorical columns
            for col in secondary_categorical:
                if col in data.columns:
                    metadata.update_column(column_name=col, sdtype='categorical')
                    
        elif dataset_name == 'africa':
            primary_categorical = ['gender', 'dropout']
            secondary_categorical = [
                'location_name', 'home_language', 'hh_occupation', 'hh_edu', 
                'hh_size', 'school_distanceKm', 'mothers_edu', 'grade', 
                'meansToSchool', 'hh_children'
            ]
            
            for col in primary_categorical + secondary_categorical:
                if col in data.columns:
                    metadata.update_column(column_name=col, sdtype='categorical')
                    if col in primary_categorical:
                        print(f"  Set {col} as primary categorical (conditional target)")
                    
        elif dataset_name == 'india':
            primary_categorical = ['STUDENTGENDER', 'STUDENT_DROPOUT_STATUS']
            secondary_categorical = [
                'GRADE', 'grade_level', 'FACULTYNAME', 'LEVELNAME', 
                'PROGRAMNAME', 'ENTRYDEGREENAME', 'ENTRY_BRANCH', 'Teacher_ID'
            ]
            
            for col in primary_categorical + secondary_categorical:
                if col in data.columns:
                    metadata.update_column(column_name=col, sdtype='categorical')
                    if col in primary_categorical:
                        print(f"  Set {col} as primary categorical (conditional target)")
        
        # Ensure numerical columns are properly typed
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in [c for c in metadata.columns.keys() if metadata.columns[c]['sdtype'] == 'categorical']:
                metadata.update_column(column_name=col, sdtype='numerical')
        
        print(f"✓ Metadata created with {len(metadata.columns)} columns for conditional generation")
        return metadata
    
    def _prepare_data_for_ctgan(self, data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Prepare data with optimized types for conditional generation."""
        
        prepared_data = data.copy()
        
        if dataset_name == 'brazil':
            # Ensure target columns have consistent types
            if 'Gender' in prepared_data.columns:
                prepared_data['Gender'] = prepared_data['Gender'].astype(int)  # Keep as int for conditioning
            if 'Target' in prepared_data.columns:
                prepared_data['Target'] = prepared_data['Target'].astype(str)
            
            # Convert other binary columns to strings for consistency
            binary_cols = [
                'Tuition fees up to date', 'Displaced', 'Educational special needs', 
                'Debtor', 'Scholarship holder', 'International', 'Daytime/evening attendance'
            ]
            for col in binary_cols:
                if col in prepared_data.columns:
                    prepared_data[col] = prepared_data[col].astype(str)
                    
        elif dataset_name == 'africa':
            categorical_cols = [
                'location_name', 'home_language', 'hh_occupation', 'hh_edu', 
                'hh_size', 'school_distanceKm', 'gender', 'mothers_edu', 
                'grade', 'meansToSchool', 'hh_children', 'dropout'
            ]
            for col in categorical_cols:
                if col in prepared_data.columns:
                    prepared_data[col] = prepared_data[col].astype(str)
                    
        elif dataset_name == 'india':
            categorical_cols = [
                'GRADE', 'grade_level', 'FACULTYNAME', 'LEVELNAME', 
                'PROGRAMNAME', 'ENTRYDEGREENAME', 'ENTRY_BRANCH', 
                'STUDENT_DROPOUT_STATUS', 'STUDENTGENDER'
            ]
            for col in categorical_cols:
                if col in prepared_data.columns:
                    prepared_data[col] = prepared_data[col].astype(str)
        
        # Handle missing values
        for col in prepared_data.columns:
            if prepared_data[col].dtype == 'object':
                prepared_data[col] = prepared_data[col].fillna('Unknown')
            else:
                prepared_data[col] = prepared_data[col].fillna(prepared_data[col].median())
        
        print(f"✓ Data prepared for conditional generation: {prepared_data.shape}")
        return prepared_data
    
    def _post_process_samples(self, samples: pd.DataFrame, spec: Dict, dataset_name: str) -> pd.DataFrame:
        """Post-process samples to ensure correct target values."""
        
        if len(samples) == 0:
            return samples
        
        # Force correct target values
        if dataset_name == 'brazil':
            samples['Gender'] = int(spec['sensitive_val'])
            samples['Target'] = str(spec['label_val'])
        elif dataset_name == 'africa':
            samples['gender'] = str(spec['sensitive_val'])
            samples['dropout'] = str(spec['label_val'])
        elif dataset_name == 'india':
            samples['STUDENTGENDER'] = str(spec['sensitive_val'])
            samples['STUDENT_DROPOUT_STATUS'] = str(spec['label_val'])
        
        # Clean up numerical values
        for col in samples.columns:
            if samples[col].dtype in ['float64', 'int64']:
                if col in ['Age at enrollment', 'age']:
                    samples[col] = samples[col].clip(lower=16, upper=40)
                elif 'GPA' in col or 'grade' in col.lower():
                    samples[col] = samples[col].clip(lower=0, upper=20)
                elif 'unemployment' in col.lower():
                    samples[col] = samples[col].clip(lower=0, upper=25)
                elif 'inflation' in col.lower():
                    samples[col] = samples[col].clip(lower=-3, upper=10)
            elif samples[col].dtype == 'object':
                samples[col] = samples[col].fillna('Unknown')
        
        return samples
    
    def _ensure_column_compatibility(self, synthetic_df: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure synthetic data matches original schema exactly."""
        
        # Add missing columns
        for col in original_data.columns:
            if col not in synthetic_df.columns:
                if original_data[col].dtype == 'object':
                    synthetic_df[col] = 'Unknown'
                else:
                    synthetic_df[col] = original_data[col].median()
        
        # Remove extra columns and reorder
        synthetic_df = synthetic_df[original_data.columns]
        
        # Match data types
        for col in original_data.columns:
            try:
                if original_data[col].dtype == 'object':
                    synthetic_df[col] = synthetic_df[col].astype(str)
                elif original_data[col].dtype in ['int64', 'int32']:
                    synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce').fillna(0).astype(int)
                else:
                    synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce').fillna(original_data[col].median())
            except:
                if original_data[col].dtype == 'object':
                    synthetic_df[col] = 'Unknown'
                else:
                    synthetic_df[col] = original_data[col].median()
        
        return synthetic_df


class TVAESyntheticGenerator:
    """
    TVAE-based generator with corrected conditional sampling.
    Note: TVAE has more limited conditional capabilities than CTGAN.
    """
    
    def __init__(self, epochs: int = 100, batch_size: int = 500, verbose: bool = True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.metadata = None
        
        if not SDV_AVAILABLE:
            raise ImportError("SDV library is required. Install with: pip install sdv")
    
    def fit_and_generate(self, original_data: pd.DataFrame, target_specs: List[Dict], 
                        dataset_name: str) -> pd.DataFrame:
        """Train TVAE and generate samples (primarily using rejection sampling)."""
        
        print(f"Training TVAE on {len(original_data)} samples for {self.epochs} epochs...")
        
        # Prepare data for TVAE
        prepared_data = self._prepare_data_for_tvae(original_data, dataset_name)
        
        # Create metadata
        self.metadata = self._create_metadata(prepared_data, dataset_name)
        
        # Initialize and train TVAE
        print("Initializing TVAE with metadata...")
        self.model = TVAESynthesizer(
            metadata=self.metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        
        print("Starting TVAE training...")
        self.model.fit(prepared_data)
        print("✓ TVAE training completed")
        
        # Generate samples (TVAE primarily uses rejection sampling due to limited conditional support)
        all_synthetic = []
        
        for spec in target_specs:
            if spec['count'] > 0:
                print(f"Generating {spec['count']} samples for {spec['category']}")
                
                # TVAE generation with conditional attempts
                synthetic_batch = self._generate_tvae_samples_with_conditions(spec, dataset_name)
                
                if len(synthetic_batch) > 0:
                    all_synthetic.append(synthetic_batch)
                    print(f"✓ Generated {len(synthetic_batch)} samples for {spec['category']}")
                else:
                    print(f"⚠ Failed to generate samples for {spec['category']}")
        
        if all_synthetic:
            synthetic_df = pd.concat(all_synthetic, ignore_index=True)
            synthetic_df = self._ensure_column_compatibility(synthetic_df, original_data)
            augmented_data = pd.concat([original_data, synthetic_df], ignore_index=True)
            print(f"✓ Created augmented dataset with {len(augmented_data)} total samples")
            return augmented_data
        else:
            print("⚠ No synthetic samples generated, returning original data")
            return original_data
    
    def _generate_tvae_samples_with_conditions(self, spec: Dict, dataset_name: str) -> pd.DataFrame:
        """Generate TVAE samples with conditional attempts and fallback to rejection sampling."""
        
        samples_needed = spec['count']
        
        # Strategy 1: Try conditional sampling (limited TVAE support)
        try:
            print(f"  Attempting TVAE conditional sampling...")
            condition_dict = self._create_condition_dict(spec, dataset_name)
            
            # Try the same methods as CTGAN
            try:
                conditions = [Condition(condition_dict, num_rows=samples_needed)]
                synthetic_batch = self.model.sample_conditions(conditions=conditions)
                
                if self._validate_conditional_results(synthetic_batch, spec, dataset_name):
                    print(f"  ✓ TVAE conditional sampling successful")
                    return self._post_process_samples(synthetic_batch, spec, dataset_name)
                    
            except Exception as e:
                print(f"  ✗ TVAE conditional sampling failed: {e}")
        
        except Exception as e:
            print(f"  ✗ TVAE conditional setup failed: {e}")
        
        # Strategy 2: Optimized rejection sampling (TVAE's strength)
        print(f"  Using optimized TVAE rejection sampling...")
        return self._tvae_rejection_sampling_optimized(spec, dataset_name, samples_needed)
    
    def _tvae_rejection_sampling_optimized(self, spec: Dict, dataset_name: str, samples_needed: int) -> pd.DataFrame:
        """Highly optimized rejection sampling for TVAE."""
        
        all_valid_samples = []
        attempt = 0
        max_attempts = 6
        
        # TVAE typically has better filtering rates than CTGAN
        estimated_success_rate = 0.25  # Start with 25% estimate for TVAE
        
        print(f"    TVAE rejection sampling for {samples_needed} samples...")
        
        while len(all_valid_samples) < samples_needed and attempt < max_attempts:
            attempt += 1
            
            remaining_needed = samples_needed - len(all_valid_samples)
            
            # Progressive batch sizing for TVAE
            if attempt <= 2:
                batch_size = min(remaining_needed * 3, 1500)  # Smaller initial batches
            else:
                batch_size = min(remaining_needed * 5, 2000)  # Larger batches if needed
            
            try:
                synthetic_batch = self.model.sample(num_rows=batch_size)
                valid_samples = self._filter_samples_strict(synthetic_batch, spec, dataset_name)
                
                if len(valid_samples) > 0:
                    all_valid_samples.extend(valid_samples.to_dict('records'))
                    actual_success_rate = len(valid_samples) / batch_size
                    estimated_success_rate = 0.6 * estimated_success_rate + 0.4 * actual_success_rate
                    
                    print(f"      Attempt {attempt}: {len(valid_samples)}/{batch_size} valid ({actual_success_rate:.1%}, total: {len(all_valid_samples)})")
                else:
                    estimated_success_rate *= 0.7
                    print(f"      Attempt {attempt}: 0/{batch_size} valid samples")
                
            except Exception as e:
                print(f"      Attempt {attempt} failed: {e}")
        
        # Process results
        if len(all_valid_samples) >= samples_needed:
            final_samples = all_valid_samples[:samples_needed]
            result_df = pd.DataFrame(final_samples)
            print(f"    ✓ TVAE rejection sampling successful: {len(final_samples)} samples")
            return self._post_process_samples(result_df, spec, dataset_name)
        
        elif len(all_valid_samples) > 0:
            # Partial success - pad with forced values
            result_df = pd.DataFrame(all_valid_samples)
            remaining = samples_needed - len(result_df)
            
            if remaining > 0:
                try:
                    padding_batch = self.model.sample(num_rows=remaining)
                    padding_batch = self._post_process_samples(padding_batch, spec, dataset_name)
                    result_df = pd.concat([result_df, padding_batch], ignore_index=True)
                except:
                    pass
            
            print(f"    ⚠ TVAE partial success: {len(result_df)} samples")
            return self._post_process_samples(result_df, spec, dataset_name)
        
        else:
            # Complete fallback
            print(f"    Fallback: Generating unconditional TVAE samples...")
            try:
                fallback_batch = self.model.sample(num_rows=samples_needed)
                return self._post_process_samples(fallback_batch, spec, dataset_name)
            except Exception as e:
                print(f"    ✗ TVAE fallback failed: {e}")
                return pd.DataFrame()
    
    # Use same helper methods as CTGAN
    def _create_condition_dict(self, spec: Dict, dataset_name: str) -> Dict:
        return CTGANSyntheticGenerator(epochs=1)._create_condition_dict(spec, dataset_name)
    
    def _validate_conditional_results(self, samples: pd.DataFrame, spec: Dict, dataset_name: str) -> bool:
        return CTGANSyntheticGenerator(epochs=1)._validate_conditional_results(samples, spec, dataset_name)
    
    def _filter_samples_strict(self, samples: pd.DataFrame, spec: Dict, dataset_name: str) -> pd.DataFrame:
        return CTGANSyntheticGenerator(epochs=1)._filter_samples_strict(samples, spec, dataset_name)
    
    def _create_metadata(self, data: pd.DataFrame, dataset_name: str) -> SingleTableMetadata:
        return CTGANSyntheticGenerator(epochs=1)._create_metadata(data, dataset_name)
    
    def _prepare_data_for_tvae(self, data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        return CTGANSyntheticGenerator(epochs=1)._prepare_data_for_ctgan(data, dataset_name)
    
    def _post_process_samples(self, samples: pd.DataFrame, spec: Dict, dataset_name: str) -> pd.DataFrame:
        return CTGANSyntheticGenerator(epochs=1)._post_process_samples(samples, spec, dataset_name)
    
    def _ensure_column_compatibility(self, synthetic_df: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        return CTGANSyntheticGenerator(epochs=1)._ensure_column_compatibility(synthetic_df, original_data)


# Updated usage functions with better epoch settings
def run_ctgan_experiment(original_data: pd.DataFrame, config: Dict, augmentation_plan: Dict) -> pd.DataFrame:
    """Run CTGAN experiment with proper conditional sampling."""
    
    # Optimized settings for conditional generation
    generator = CTGANSyntheticGenerator(epochs=100, batch_size=500, verbose=True)
    
    # Prepare target specs
    breakdown = augmentation_plan['breakdown']
    target_specs = []
    
    if config['dataname'] == 'brazil':
        mapping = {'priv_pos': (1, 'Dropout'), 'priv_neg': (1, 'Graduate'), 
                  'unpriv_pos': (0, 'Dropout'), 'unpriv_neg': (0, 'Graduate')}
    elif config['dataname'] == 'africa':
        mapping = {'priv_pos': ('Male', 'Yes'), 'priv_neg': ('Male', 'No'), 
                  'unpriv_pos': ('Female', 'Yes'), 'unpriv_neg': ('Female', 'No')}
    elif config['dataname'] == 'india':
        mapping = {'priv_pos': ('M', 'DROPOUT'), 'priv_neg': ('M', 'NOT DROPOUT'), 
                  'unpriv_pos': ('F', 'DROPOUT'), 'unpriv_neg': ('F', 'NOT DROPOUT')}
    else:
        raise ValueError(f"Unsupported dataset: {config['dataname']}")
    
    for category, count in breakdown.items():
        if count > 0:
            sensitive_val, label_val = mapping[category]
            target_specs.append({
                'category': category,
                'count': count,
                'sensitive_val': sensitive_val,
                'label_val': label_val
            })
    
    return generator.fit_and_generate(original_data, target_specs, config['dataname'])


def run_tvae_experiment(original_data: pd.DataFrame, config: Dict, augmentation_plan: Dict) -> pd.DataFrame:
    """Run TVAE experiment with proper conditional attempts."""
    
    # TVAE settings optimized for rejection sampling
    generator = TVAESyntheticGenerator(epochs=75, batch_size=500, verbose=True)
    
    # Same target spec preparation as CTGAN
    breakdown = augmentation_plan['breakdown']
    target_specs = []
    
    if config['dataname'] == 'brazil':
        mapping = {'priv_pos': (1, 'Dropout'), 'priv_neg': (1, 'Graduate'), 
                  'unpriv_pos': (0, 'Dropout'), 'unpriv_neg': (0, 'Graduate')}
    elif config['dataname'] == 'africa':
        mapping = {'priv_pos': ('Male', 'Yes'), 'priv_neg': ('Male', 'No'), 
                  'unpriv_pos': ('Female', 'Yes'), 'unpriv_neg': ('Female', 'No')}
    elif config['dataname'] == 'india':
        mapping = {'priv_pos': ('M', 'DROPOUT'), 'priv_neg': ('M', 'NOT DROPOUT'), 
                  'unpriv_pos': ('F', 'DROPOUT'), 'unpriv_neg': ('F', 'NOT DROPOUT')}
    else:
        raise ValueError(f"Unsupported dataset: {config['dataname']}")
    
    for category, count in breakdown.items():
        if count > 0:
            sensitive_val, label_val = mapping[category]
            target_specs.append({
                'category': category,
                'count': count,
                'sensitive_val': sensitive_val,
                'label_val': label_val
            })
    
    return generator.fit_and_generate(original_data, target_specs, config['dataname'])