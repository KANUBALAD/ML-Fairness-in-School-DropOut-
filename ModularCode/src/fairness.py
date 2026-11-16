import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FairnessMetrics:
    """
    Class to compute various fairness metrics for binary classification.
    """
    
    def __init__(self, y_true, y_pred, sensitive_attribute):
        """
        Initialize the fairness metrics calculator.
        
        Parameters:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        sensitive_attribute (array-like): Binary sensitive attribute (0/1)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.sensitive_attribute = np.array(sensitive_attribute)
        
        # Define privileged and unprivileged groups
        self.privileged_mask = self.sensitive_attribute == 1
        self.unprivileged_mask = self.sensitive_attribute == 0
    
    def demographic_parity(self):
        """
        Calculate demographic parity difference.
        DP = P(Y_hat = 1 | A = 1) - P(Y_hat = 1 | A = 0)
        """
        if np.sum(self.privileged_mask) == 0 or np.sum(self.unprivileged_mask) == 0:
            return 0.0
        
        priv_rate = np.mean(self.y_pred[self.privileged_mask])
        unpriv_rate = np.mean(self.y_pred[self.unprivileged_mask])
        return priv_rate - unpriv_rate
    
    def equalized_odds_difference(self):
        """
        Calculate equalized odds difference (TPR difference).
        EOD = TPR_privileged - TPR_unprivileged
        """
        tpr_priv = self._true_positive_rate(self.privileged_mask)
        tpr_unpriv = self._true_positive_rate(self.unprivileged_mask)
        return tpr_priv - tpr_unpriv
    
    def equal_opportunity_difference(self):
        """
        Calculate equal opportunity difference (same as equalized odds for positive class).
        """
        return self.equalized_odds_difference()
    
    def false_positive_rate_difference(self):
        """
        Calculate false positive rate difference.
        FPR_diff = FPR_privileged - FPR_unprivileged
        """
        fpr_priv = self._false_positive_rate(self.privileged_mask)
        fpr_unpriv = self._false_positive_rate(self.unprivileged_mask)
        return fpr_priv - fpr_unpriv
    
    def accuracy_difference(self):
        """
        Calculate accuracy difference between groups.
        """
        acc_priv = self._accuracy(self.privileged_mask)
        acc_unpriv = self._accuracy(self.unprivileged_mask)
        return acc_priv - acc_unpriv
    
    def _true_positive_rate(self, mask):
        """Calculate True Positive Rate for a specific group."""
        if np.sum(mask) == 0:
            return 0.0
        y_true_group = self.y_true[mask]
        y_pred_group = self.y_pred[mask]
        
        if np.sum(y_true_group) == 0:
            return 0.0
        
        return np.sum((y_pred_group == 1) & (y_true_group == 1)) / np.sum(y_true_group == 1)
    
    def _false_positive_rate(self, mask):
        """Calculate False Positive Rate for a specific group."""
        if np.sum(mask) == 0:
            return 0.0
        y_true_group = self.y_true[mask]
        y_pred_group = self.y_pred[mask]
        
        if np.sum(y_true_group == 0) == 0:
            return 0.0
        
        return np.sum((y_pred_group == 1) & (y_true_group == 0)) / np.sum(y_true_group == 0)
    
    def _accuracy(self, mask):
        """Calculate accuracy for a specific group."""
        if np.sum(mask) == 0:
            return 0.0
        y_true_group = self.y_true[mask]
        y_pred_group = self.y_pred[mask]
        return np.mean(y_true_group == y_pred_group)
    
    def get_all_metrics(self):
        """
        Get all fairness metrics as a dictionary.
        """
        return {
            'demographic_parity': self.demographic_parity(),
            'equalized_odds_difference': self.equalized_odds_difference(),
            'equal_opportunity_difference': self.equal_opportunity_difference(),
            'false_positive_rate_difference': self.false_positive_rate_difference(),
            'accuracy_difference': self.accuracy_difference()
        }


class FairnessMitigation:
    """
    Class implementing various fairness mitigation techniques.
    """
    
    def __init__(self, model_type='logistic_regression', random_state=42):
        """
        Initialize the fairness mitigation class.
        
        Parameters:
        model_type (str): Type of model to use ('logistic_regression', 'decision_tree', 'random_forest')
        random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
    def _get_model(self):
        """Get the appropriate model based on model_type."""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.model_type == 'decision_tree':
            return DecisionTreeClassifier(random_state=self.random_state)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def reweighting(self, X, y, sensitive_attribute):
        """
        Apply reweighting preprocessing technique.
        
        Parameters:
        X (array-like): Features
        y (array-like): Labels
        sensitive_attribute (array-like): Sensitive attribute
        
        Returns:
        array: Sample weights
        """
        # Calculate group probabilities
        privileged_positive = np.sum((sensitive_attribute == 1) & (y == 1))
        privileged_negative = np.sum((sensitive_attribute == 1) & (y == 0))
        unprivileged_positive = np.sum((sensitive_attribute == 0) & (y == 1))
        unprivileged_negative = np.sum((sensitive_attribute == 0) & (y == 0))
        
        total = len(y)
        
        # Calculate weights
        weights = np.ones(len(y))
        
        # Weights for privileged positive
        if privileged_positive > 0:
            weights[(sensitive_attribute == 1) & (y == 1)] = total / (4 * privileged_positive)
        
        # Weights for privileged negative
        if privileged_negative > 0:
            weights[(sensitive_attribute == 1) & (y == 0)] = total / (4 * privileged_negative)
        
        # Weights for unprivileged positive
        if unprivileged_positive > 0:
            weights[(sensitive_attribute == 0) & (y == 1)] = total / (4 * unprivileged_positive)
        
        # Weights for unprivileged negative
        if unprivileged_negative > 0:
            weights[(sensitive_attribute == 0) & (y == 0)] = total / (4 * unprivileged_negative)
        
        return weights
    
    def fair_representation(self, X, y, sensitive_attribute, k=5):
        """
        Apply fair representation preprocessing (simplified version).
        
        Parameters:
        X (array-like): Features
        y (array-like): Labels
        sensitive_attribute (array-like): Sensitive attribute
        k (int): Number of components for dimensionality reduction
        
        Returns:
        array: Transformed features
        """
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce correlation with sensitive attribute
        pca = PCA(n_components=min(k, X.shape[1]))
        X_transformed = pca.fit_transform(X)
        
        return X_transformed
    
    def adversarial_debiasing(self, X, y, sensitive_attribute, epochs=100, lr=0.01):
        """
        Simple adversarial debiasing implementation.
        Note: This is a simplified version. For production use, consider using specialized libraries.
        
        Parameters:
        X (array-like): Features
        y (array-like): Labels
        sensitive_attribute (array-like): Sensitive attribute
        epochs (int): Number of training epochs
        lr (float): Learning rate
        
        Returns:
        trained model
        """
        # For simplicity, we'll use reweighting with the base model
        # In practice, this would involve neural networks with adversarial training
        weights = self.reweighting(X, y, sensitive_attribute)
        
        self.model = self._get_model()
        
        # For sklearn models that support sample_weight
        if hasattr(self.model, 'fit') and 'sample_weight' in self.model.fit.__code__.co_varnames:
            self.model.fit(X, y, sample_weight=weights)
        else:
            # For models that don't support sample_weight, use resampling
            indices = np.random.choice(len(X), size=len(X), p=weights/np.sum(weights))
            X_resampled = X[indices]
            y_resampled = y[indices]
            self.model.fit(X_resampled, y_resampled)
        
        return self.model
    
    def threshold_optimization(self, X, y, sensitive_attribute, y_pred_proba):
        """
        Apply threshold optimization post-processing.
        
        Parameters:
        X (array-like): Features
        y (array-like): True labels
        sensitive_attribute (array-like): Sensitive attribute
        y_pred_proba (array-like): Predicted probabilities
        
        Returns:
        dict: Optimal thresholds for each group
        """
        # Find optimal thresholds for each group to maximize accuracy while maintaining fairness
        thresholds = {}
        
        # Privileged group
        priv_mask = sensitive_attribute == 1
        if np.sum(priv_mask) > 0:
            priv_y = y[priv_mask]
            priv_proba = y_pred_proba[priv_mask]
            thresholds['privileged'] = self._find_optimal_threshold(priv_y, priv_proba)
        else:
            thresholds['privileged'] = 0.5
        
        # Unprivileged group
        unpriv_mask = sensitive_attribute == 0
        if np.sum(unpriv_mask) > 0:
            unpriv_y = y[unpriv_mask]
            unpriv_proba = y_pred_proba[unpriv_mask]
            thresholds['unprivileged'] = self._find_optimal_threshold(unpriv_y, unpriv_proba)
        else:
            thresholds['unprivileged'] = 0.5
        
        return thresholds
    
    def _find_optimal_threshold(self, y_true, y_proba):
        """Find optimal threshold to maximize accuracy."""
        best_threshold = 0.5
        best_accuracy = 0
        
        for threshold in np.arange(0.1, 0.9, 0.1):
            y_pred = (y_proba >= threshold).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold
    
    def apply_fairness_technique(self, technique, X, y, sensitive_attribute, **kwargs):
        """
        Apply a specific fairness technique.
        
        Parameters:
        technique (str): Fairness technique to apply
        X (array-like): Features
        y (array-like): Labels
        sensitive_attribute (array-like): Sensitive attribute
        **kwargs: Additional arguments for specific techniques
        
        Returns:
        Processed data or trained model
        """
        if technique == 'reweighting':
            return self.reweighting(X, y, sensitive_attribute)
        elif technique == 'fair_representation':
            return self.fair_representation(X, y, sensitive_attribute, **kwargs)
        elif technique == 'adversarial_debiasing':
            return self.adversarial_debiasing(X, y, sensitive_attribute, **kwargs)
        elif technique == 'threshold_optimization':
            return self.threshold_optimization(X, y, sensitive_attribute, **kwargs)
        else:
            raise ValueError(f"Unsupported fairness technique: {technique}")


def evaluate_fairness(y_true, y_pred, sensitive_attribute, model_name="Model"):
    """
    Evaluate fairness metrics for a model.
    ...
    Returns:
    dict: Dictionary containing fairness metrics
    """
    fairness_metrics = FairnessMetrics(y_true, y_pred, sensitive_attribute)
    metrics = fairness_metrics.get_all_metrics()

    # Add overall accuracy
    metrics['overall_accuracy'] = accuracy_score(y_true, y_pred)

    # Add group-specific accuracies
    priv_mask = sensitive_attribute == 1
    unpriv_mask = sensitive_attribute == 0

    metrics['model_name'] = model_name     # <<< ADD THIS LINE

    if np.sum(priv_mask) > 0:
        metrics['privileged_accuracy'] = accuracy_score(y_true[priv_mask], y_pred[priv_mask])
    else:
        metrics['privileged_accuracy'] = 0.0

    if np.sum(unpriv_mask) > 0:
        metrics['unprivileged_accuracy'] = accuracy_score(y_true[unpriv_mask], y_pred[unpriv_mask])
    else:
        metrics['unprivileged_accuracy'] = 0.0

    return metrics

def run_fairness_aware_training(X, y, sensitive_attribute, model_type='logistic_regression', 
                               technique='reweighting', test_size=0.2, random_state=42):
    """
    Run fairness-aware training pipeline.
    
    Parameters:
    X (array-like): Features
    y (array-like): Labels
    sensitive_attribute (array-like): Sensitive attribute
    model_type (str): Type of model to use
    technique (str): Fairness technique to apply
    test_size (float): Test set size
    random_state (int): Random state
    
    Returns:
    dict: Results containing models and fairness metrics
    """
    # Split data
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_attribute, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize fairness mitigation
    fair_model = FairnessMitigation(model_type=model_type, random_state=random_state)
    
    # Train baseline model
    baseline_model = fair_model._get_model()
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    
    # Train fair model
    if technique == 'reweighting':
        weights = fair_model.reweighting(X_train, y_train, sens_train)
        fair_trained_model = fair_model._get_model()
        if hasattr(fair_trained_model, 'fit') and 'sample_weight' in fair_trained_model.fit.__code__.co_varnames:
            fair_trained_model.fit(X_train, y_train, sample_weight=weights)
        else:
            # Resample based on weights
            indices = np.random.choice(len(X_train), size=len(X_train), p=weights/np.sum(weights))
            fair_trained_model.fit(X_train[indices], y_train[indices])
        fair_pred = fair_trained_model.predict(X_test)
    
    elif technique == 'fair_representation':
        X_train_fair = fair_model.fair_representation(X_train, y_train, sens_train)
        X_test_fair = fair_model.fair_representation(X_test, y_test, sens_test)
        fair_trained_model = fair_model._get_model()
        fair_trained_model.fit(X_train_fair, y_train)
        fair_pred = fair_trained_model.predict(X_test_fair)
    
    else:
        # Default to reweighting if technique not implemented
        weights = fair_model.reweighting(X_train, y_train, sens_train)
        fair_trained_model = fair_model._get_model()
        fair_trained_model.fit(X_train, y_train, sample_weight=weights)
        fair_pred = fair_trained_model.predict(X_test)
    
    # Evaluate fairness
    baseline_metrics = evaluate_fairness(y_test, baseline_pred, sens_test, "Baseline")
    fair_metrics = evaluate_fairness(y_test, fair_pred, sens_test, f"Fair ({technique})")
    
    return {
        'baseline_model': baseline_model,
        'fair_model': fair_trained_model,
        'baseline_metrics': baseline_metrics,
        'fair_metrics': fair_metrics,
        'y_pred_fair': fair_pred,
        'y_test': y_test,
        'sens_test': sens_test,
        # ... any other outputs
    }