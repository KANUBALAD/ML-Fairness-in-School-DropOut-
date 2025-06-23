import re
import pandas as pd
import numpy as np
from faker import Faker
import json
import random
from typing import Dict, List, Tuple, Optional
import time

class SyntheticDataGenerator:
    """
    Faker-based synthetic data generator for fairness research.
    Generates balanced/imbalanced datasets to study fairness intervention effectiveness.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the synthetic data generator.
        
        Parameters:
        api_key (str): Not needed anymore (kept for compatibility)
        base_url (str): Not needed anymore (kept for compatibility)
        """
        self.faker = Faker(['en_US', 'pt_BR', 'hi_IN'])  # Multi-locale support
        
        # Dataset schemas with enhanced generation rules
        self.schemas = {
            'africa': {
                'columns': ['location_name', 'home_language', 'hh_occupation', 'hh_edu', 
                           'hh_size', 'school_distanceKm', 'age', 'gender', 'mothers_edu', 
                           'grade', 'meansToSchool', 'hh_children', 'dropout'],
                'generation_rules': {
                    'location_name': ['Rural', 'Urban', 'Semi-urban'],
                    'home_language': ['English', 'Kiswahili', 'Local dialect'],
                    'hh_occupation': ['Unemployed', 'Self-employed', 'Agriculture', 'Government', 'Private'],
                    'hh_edu': ['Primary', 'Secondary', 'Tertiary', 'None'],
                    'hh_size': ['Less than three', 'Three to five', 'More than five'],
                    'school_distanceKm': ['Less than 1 km', '1-2 km', '2-3 km', '4-5 km', '6-10 km', 'More than 11 km'],
                    'mothers_edu': ['Primary', 'Secondary', 'Tertiary', 'None'],
                    'grade': ['Form One', 'Form Two', 'Form Three', 'Form Four'],
                    'meansToSchool': ['Walk', 'Bicycle', 'Public transport', 'Private transport'],
                    'hh_children': ['One Child', 'Two Children', 'Three Children', 'Four Children', 'Five Children', 'More than five']
                }
            },
            'brazil': {
                'columns': ['Marital status', 'Application mode', 'Application order', 'Course',
                           'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
                           'Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation',
                           'Father\'s occupation', 'Displaced', 'Educational special needs', 'Debtor',
                           'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment',
                           'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
                           'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
                           'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
                           'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
                           'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
                           'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
                           'Unemployment rate', 'Inflation rate', 'GDP', 'Target'],
                'generation_rules': {
                    'courses': list(range(1, 21)),
                    'application_modes': list(range(1, 21)),
                    'qualifications': list(range(1, 41)),
                    'occupations': list(range(0, 21))
                }
            },
            'india': {
                'columns': ['GRADE', 'grade_level', 'Teacher_ID', 'STUDENTCODE', 'entry_GPA',
                           'FACULTYNAME', 'LEVELNAME', 'PROGRAMNAME', 'STUDENT_ENTRY_YEAR',
                           'ENTRYDEGREENAME', 'SCHOOLNAME', 'SCHOOL_PROVINCENAME', 'ENTRY_BRANCH',
                           'STUDENT_DROPOUT_STATUS', 'BIRTHDATE', 'STUDENTGENDER'],
                'generation_rules': {
                    'GRADE': ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F'],
                    'grade_level': ['1-LOW', '2-MEDIUM', '3-HIGH'],
                    'FACULTYNAME': ['Integrated medicine', 'Engineering', 'Business', 'Arts', 'Science'],
                    'LEVELNAME': ['Bachelor\'s degree', 'Master\'s degree', 'Diploma'],
                    'PROGRAMNAME': ['Health and beauty', 'Applied Thai Traditional Medicine', 'Computer Science', 'Business Administration'],
                    'ENTRYDEGREENAME': ['Mattayom 6', 'Diploma', 'Bachelor'],
                    'ENTRY_BRANCH': ['Wit-Mathematics', 'Wit-Science', 'Wit-Arts', 'General'],
                    'provinces': ['Bangkok', 'Lop Buri', 'Chachoengsao', 'Chiang Mai', 'Phuket', 'Khon Kaen']
                }
            }
        }

    def generate_category_specific_samples(self, dataset_name: str, n_samples: int, category: str, reference_data: pd.DataFrame) -> List[Dict]:
        """
        Generate samples for a specific category using robust Faker-based generation.
        """
        
        # Define category mappings
        category_mapping = {
            'africa': {
                'sensitive_col': 'gender',
                'label_col': 'dropout',
                'privileged_val': 'Male',
                'unprivileged_val': 'Female',
                'positive_val': 'Yes',
                'negative_val': 'No'
            },
            'brazil': {
                'sensitive_col': 'Gender',
                'label_col': 'Target',
                'privileged_val': 1,  # Male
                'unprivileged_val': 0,  # Female
                'positive_val': 'Dropout',
                'negative_val': 'Graduate'
            },
            'india': {
                'sensitive_col': 'STUDENTGENDER',
                'label_col': 'STUDENT_DROPOUT_STATUS',
                'privileged_val': 'M',
                'unprivileged_val': 'F',
                'positive_val': 'DROPOUT',
                'negative_val': 'NOT DROPOUT'
            }
        }
        
        mapping = category_mapping[dataset_name]
        
        # Determine target values based on category
        if category == 'priv_pos':
            target_sensitive = mapping['privileged_val']
            target_label = mapping['positive_val']
            description = "privileged group with positive outcome"
        elif category == 'priv_neg':
            target_sensitive = mapping['privileged_val']
            target_label = mapping['negative_val']
            description = "privileged group with negative outcome"
        elif category == 'unpriv_pos':
            target_sensitive = mapping['unprivileged_val']
            target_label = mapping['positive_val']
            description = "unprivileged group with positive outcome"
        elif category == 'unpriv_neg':
            target_sensitive = mapping['unprivileged_val']
            target_label = mapping['negative_val']
            description = "unprivileged group with negative outcome"
        else:
            raise ValueError(f"Unknown category: {category}")
        
        print(f"  Generating {n_samples} samples for {description}")
        print(f"    Using 100% Faker generation for reliability and speed")
        
        # Generate all samples using enhanced Faker
        all_samples = self._generate_enhanced_faker_samples(
            dataset_name, n_samples, target_sensitive, target_label, mapping, reference_data
        )
        
        print(f"    ✓ Generated {len(all_samples)} samples")
        return all_samples

    def _generate_enhanced_faker_samples(self, dataset_name: str, n_samples: int, 
                                       target_sensitive, target_label, mapping: Dict, 
                                       reference_data: pd.DataFrame) -> List[Dict]:
        """Generate realistic samples using enhanced Faker with statistical patterns from reference data."""
        
        samples = []
        
        # Analyze reference data for realistic patterns
        patterns = self._analyze_reference_patterns(reference_data, dataset_name)
        
        for i in range(n_samples):
            if dataset_name == 'africa':
                sample = self._generate_africa_sample(target_sensitive, target_label, patterns)
            elif dataset_name == 'brazil':
                sample = self._generate_brazil_sample(target_sensitive, target_label, patterns)
            elif dataset_name == 'india':
                sample = self._generate_india_sample(target_sensitive, target_label, patterns)
            
            samples.append(sample)
        
        return samples

    def _analyze_reference_patterns(self, reference_data: pd.DataFrame, dataset_name: str) -> Dict:
        """Analyze reference data to extract realistic patterns for generation."""
        
        patterns = {}
        
        try:
            if dataset_name == 'brazil':
                # Analyze numerical column ranges
                patterns['age_range'] = (
                    max(15, int(reference_data['Age at enrollment'].min())),
                    min(50, int(reference_data['Age at enrollment'].max()))
                )
                patterns['unemployment_range'] = (
                    reference_data['Unemployment rate'].min(),
                    reference_data['Unemployment rate'].max()
                )
                patterns['inflation_range'] = (
                    reference_data['Inflation rate'].min(),
                    reference_data['Inflation rate'].max()
                )
                patterns['gdp_range'] = (
                    reference_data['GDP'].min(),
                    reference_data['GDP'].max()
                )
                
            elif dataset_name == 'africa':
                patterns['age_range'] = (11, 18)
                
            elif dataset_name == 'india':
                patterns['gpa_range'] = (2.0, 4.0)
                patterns['entry_years'] = [2018, 2019, 2020, 2021, 2022, 2023]
                
        except Exception as e:
            print(f"    Pattern analysis failed: {e}, using defaults")
            # Set default patterns
            patterns = self._get_default_patterns(dataset_name)
        
        return patterns

    def _get_default_patterns(self, dataset_name: str) -> Dict:
        """Get default patterns when reference analysis fails."""
        
        if dataset_name == 'brazil':
            return {
                'age_range': (18, 35),
                'unemployment_range': (5.0, 15.0),
                'inflation_range': (-2.0, 8.0),
                'gdp_range': (-5.0, 5.0)
            }
        elif dataset_name == 'africa':
            return {'age_range': (11, 18)}
        elif dataset_name == 'india':
            return {
                'gpa_range': (2.0, 4.0),
                'entry_years': [2018, 2019, 2020, 2021, 2022, 2023]
            }
        return {}

    def _generate_africa_sample(self, target_sensitive, target_label, patterns: Dict) -> Dict:
        """Generate a realistic Africa dataset sample."""
        
        age_min, age_max = patterns.get('age_range', (11, 18))
        
        return {
            'location_name': random.choice(['Rural', 'Urban', 'Semi-urban']),
            'home_language': random.choice(['English', 'Kiswahili', 'Local dialect']),
            'hh_occupation': random.choice(['Unemployed', 'Self-employed', 'Agriculture', 'Government', 'Private']),
            'hh_edu': random.choice(['Primary', 'Secondary', 'Tertiary', 'None']),
            'hh_size': random.choice(['Less than three', 'Three to five', 'More than five']),
            'school_distanceKm': random.choice(['Less than 1 km', '1-2 km', '2-3 km', '4-5 km', '6-10 km', 'More than 11 km']),
            'age': random.randint(age_min, age_max),
            'gender': target_sensitive,
            'mothers_edu': random.choice(['Primary', 'Secondary', 'Tertiary', 'None']),
            'grade': random.choice(['Form One', 'Form Two', 'Form Three', 'Form Four']),
            'meansToSchool': random.choice(['Walk', 'Bicycle', 'Public transport', 'Private transport']),
            'hh_children': random.choice(['One Child', 'Two Children', 'Three Children', 'Four Children', 'Five Children', 'More than five']),
            'dropout': target_label
        }

    def _generate_brazil_sample(self, target_sensitive, target_label, patterns: Dict) -> Dict:
        """Generate a realistic Brazil dataset sample."""
        
        age_min, age_max = patterns.get('age_range', (18, 35))
        unemp_min, unemp_max = patterns.get('unemployment_range', (5.0, 15.0))
        inf_min, inf_max = patterns.get('inflation_range', (-2.0, 8.0))
        gdp_min, gdp_max = patterns.get('gdp_range', (-5.0, 5.0))
        
        # Generate realistic academic performance
        enrolled = random.randint(4, 8)
        evaluations = random.randint(0, enrolled + 2)
        approved = random.randint(0, min(evaluations, enrolled))
        
        return {
            'Marital status': random.randint(1, 4),
            'Application mode': random.randint(1, 20),
            'Application order': random.randint(1, 9),
            'Course': random.randint(1, 20),
            'Daytime/evening attendance': random.randint(0, 1),
            'Previous qualification': random.randint(1, 30),
            'Nacionality': 1,
            'Mother\'s qualification': random.randint(1, 40),
            'Father\'s qualification': random.randint(1, 40),
            'Mother\'s occupation': random.randint(0, 20),
            'Father\'s occupation': random.randint(0, 20),
            'Displaced': random.randint(0, 1),
            'Educational special needs': 0,
            'Debtor': random.randint(0, 1),
            'Tuition fees up to date': random.randint(0, 1),
            'Gender': target_sensitive,
            'Scholarship holder': random.randint(0, 1),
            'Age at enrollment': random.randint(age_min, age_max),
            'International': random.randint(0, 1),
            'Curricular units 1st sem (credited)': random.randint(0, 6),
            'Curricular units 1st sem (enrolled)': enrolled,
            'Curricular units 1st sem (evaluations)': evaluations,
            'Curricular units 1st sem (approved)': approved,
            'Curricular units 1st sem (grade)': round(random.uniform(0, 20), 2),
            'Curricular units 1st sem (without evaluations)': max(0, enrolled - evaluations),
            'Curricular units 2nd sem (credited)': random.randint(0, 6),
            'Curricular units 2nd sem (enrolled)': random.randint(4, 8),
            'Curricular units 2nd sem (evaluations)': random.randint(0, 10),
            'Curricular units 2nd sem (approved)': random.randint(0, 8),
            'Curricular units 2nd sem (grade)': round(random.uniform(0, 20), 2),
            'Curricular units 2nd sem (without evaluations)': random.randint(0, 3),
            'Unemployment rate': round(random.uniform(unemp_min, unemp_max), 1),
            'Inflation rate': round(random.uniform(inf_min, inf_max), 1),
            'GDP': round(random.uniform(gdp_min, gdp_max), 2),
            'Target': target_label
        }

    def _generate_india_sample(self, target_sensitive, target_label, patterns: Dict) -> Dict:
        """Generate a realistic India dataset sample."""
        
        gpa_min, gpa_max = patterns.get('gpa_range', (2.0, 4.0))
        entry_years = patterns.get('entry_years', [2018, 2019, 2020, 2021, 2022, 2023])
        
        return {
            'GRADE': random.choice(['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'F']),
            'grade_level': random.choice(['1-LOW', '2-MEDIUM', '3-HIGH']),
            'Teacher_ID': f"Teacher{random.randint(10, 99):02d}",
            'STUDENTCODE': f"Student{random.randint(1000, 9999):04d}",
            'entry_GPA': round(random.uniform(gpa_min, gpa_max), 2),
            'FACULTYNAME': random.choice(['Integrated medicine', 'Engineering', 'Business', 'Arts', 'Science']),
            'LEVELNAME': random.choice(['Bachelor\'s degree', 'Master\'s degree', 'Diploma']),
            'PROGRAMNAME': random.choice(['Health and beauty', 'Applied Thai Traditional Medicine', 'Computer Science', 'Business Administration']),
            'STUDENT_ENTRY_YEAR': random.choice(entry_years),
            'ENTRYDEGREENAME': random.choice(['Mattayom 6', 'Diploma', 'Bachelor']),
            'SCHOOLNAME': f"School {random.randint(1, 100)}",
            'SCHOOL_PROVINCENAME': random.choice(['Bangkok', 'Lop Buri', 'Chachoengsao', 'Chiang Mai', 'Phuket', 'Khon Kaen']),
            'ENTRY_BRANCH': random.choice(['Wit-Mathematics', 'Wit-Science', 'Wit-Arts', 'General']),
            'STUDENT_DROPOUT_STATUS': target_label,
            'BIRTHDATE': self.faker.date_of_birth(minimum_age=18, maximum_age=25).strftime('%Y-%m-%d %H:%M:%S'),
            'STUDENTGENDER': target_sensitive
        }

    def _generate_default_column(self, col_name: str, dataset_name: str, n_samples: int):
        """Generate default values for missing columns."""
        
        if 'date' in col_name.lower() or 'birth' in col_name.lower():
            return [self.faker.date_of_birth(minimum_age=18, maximum_age=25).strftime('%Y-%m-%d %H:%M:%S') 
                   for _ in range(n_samples)]
        elif 'id' in col_name.lower() or 'code' in col_name.lower():
            return [f"{col_name}{i:04d}" for i in range(n_samples)]
        elif 'gpa' in col_name.lower() or 'grade' in col_name.lower():
            return [round(random.uniform(2.0, 4.0), 2) for _ in range(n_samples)]
        elif 'year' in col_name.lower():
            return [random.randint(2018, 2023) for _ in range(n_samples)]
        elif col_name.lower() in ['unemployment rate', 'inflation rate']:
            return [round(random.uniform(5.0, 15.0), 1) for _ in range(n_samples)]
        elif col_name.lower() == 'gdp':
            return [round(random.uniform(-5.0, 5.0), 2) for _ in range(n_samples)]
        else:
            return ['Unknown'] * n_samples

    # Keep other methods for compatibility (they won't be used)
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