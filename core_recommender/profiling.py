import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional
from core_recommender.logger import get_logger

logger = get_logger(__name__)

class DataProfiler:
    """Analyzes datasets to provide intelligent recommendations for model selection and configuration.

    The profiler assesses data characteristics such as normality, sparsity, linearity,
    and class imbalance to generate actionable suggestions for the modeling pipeline.

    Attributes:
        suggestions (Dict[str, Any]): A dictionary containing list of messages, models to exclude/prioritize.
        profile (Dict[str, Any]): A dictionary containing calculated statistical metrics of the dataset.
    """
    
    def __init__(self) -> None:
        """Initializes the DataProfiler."""
        self.suggestions: Dict[str, Any] = {}
        self.profile: Dict[str, Any] = {}

    def analyze(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Performs a comprehensive analysis of the dataset.

        Args:
            X (pd.DataFrame): The input features DataFrame.
            y (pd.Series): The target variable Series.

        Returns:
            Dict[str, Any]: A dictionary containing two keys:
                - 'profile': Statistical metrics (sparsity, normality, etc.).
                - 'suggestions': Recommended actions, model inclusions/exclusions.
        """
        logger.info("🔍 Intelligent Profiler: Analyzing dataset characteristics...")
        
        self.profile = {}
        self.suggestions = {
            'priority_models': [],
            'exclude_models': [],
            'config_overrides': {},
            'messages': []
        }
        
        # 1. Basic Stats
        n_samples, n_features = X.shape
        self.profile['n_samples'] = n_samples
        self.profile['n_features'] = n_features
        # Avoid division by zero
        self.profile['ratio'] = n_samples / max(n_features, 1)
        
        # 2. Sparsity Check
        numeric_X = X.select_dtypes(include=np.number)
        if not numeric_X.empty:
            zero_count = (numeric_X == 0).sum().sum()
            total_elements = numeric_X.size
            sparsity = zero_count / total_elements
            self.profile['sparsity'] = sparsity
        else:
            self.profile['sparsity'] = 0.0

        # 3. Normality Check (Sampled)
        is_gaussian = False
        if not numeric_X.empty:
            # Check first 3 features or random 3
            sample_feats = numeric_X.sample(n=min(3, numeric_X.shape[1]), axis=1, random_state=42)
            p_values = []
            for col in sample_feats.columns:
                # Shapiro test requires sample size < 5000 usually, let's take a sample
                # Drop NA to avoid errors in Shapiro
                sample_data = sample_feats[col].dropna().sample(n=min(500, len(sample_feats)), random_state=42)
                if sample_data.nunique() > 1: # Shapiro requires at least 3 unique values ideally, but definitely > 1
                    try:
                        _, p = stats.shapiro(sample_data)
                        p_values.append(p)
                    except Exception:
                        pass
            
            # If all p-values > 0.05, we might say it's roughly Gaussian (very simplified)
            if p_values and all(p > 0.05 for p in p_values):
                is_gaussian = True
        
        self.profile['is_gaussian'] = is_gaussian

        # 4. Linearity Check (Regression Only mostly, but useful hint)
        linearity_score = 0.0
        max_linear_score = 0.0
        if pd.api.types.is_numeric_dtype(y) and not numeric_X.empty:
            # Handle potential NaNs in y before corr
            valid_idx = y.dropna().index
            # Intersect indices
            common_idx = X.index.intersection(valid_idx)
            
            if not common_idx.empty:
                X_valid = numeric_X.loc[common_idx]
                y_valid = y.loc[common_idx]
                
                # Compute correlation of each feature with target
                corrs = X_valid.corrwith(y_valid).abs()
                if not corrs.empty:
                    linearity_score = corrs.mean()
                    max_linear_score = corrs.max()

        self.profile['linearity_score'] = linearity_score
        self.profile['max_linear_score'] = max_linear_score
        
        # --- GENERATE SUGGESTIONS ---
        self._generate_rules(y)
        
        logger.info(f"📊 Profile Summary: {self.profile}")
        logger.info(f"💡 Suggestions: {self.suggestions['messages']}")
        
        return {
            'profile': self.profile,
            'suggestions': self.suggestions
        }

    def _generate_rules(self, y: pd.Series) -> None:
        """Applies heuristic rules based on the profile to generate suggestions.
        
        Args:
            y (pd.Series): The target variable, used for class balance checks.
        """
        
        # Rule 1: Dataset Size vs Complexity
        if self.profile.get('n_samples', 0) > 20000:
            msg = "Large dataset detected (>20k samples). Suggesting exclusion of slow models (SVM-RBF, KNN)."
            self.suggestions['messages'].append(msg)
            self.suggestions['exclude_models'].extend(['SVM', 'KNN'])
            
        # Rule 2: High Dimensionality -> Regularization
        # If n_features > n_samples (or close)
        if self.profile.get('n_features', 0) > self.profile.get('n_samples', 0):
            msg = "High dimensionality (p > n) detected. Prioritizing Regularized Linear Models (Lasso/Ridge)."
            self.suggestions['messages'].append(msg)
            self.suggestions['priority_models'].extend(['Linear Regression', 'Logistic Regression'])
            
        # Rule 3: Gaussian Distribution
        if self.profile.get('is_gaussian'):
            msg = "Features appear normally distributed. Gaussian Naive Bayes is a strong candidate."
            self.suggestions['messages'].append(msg)
            self.suggestions['priority_models'].append('Naive Bayes')
            
        # Rule 4: Linearity (for Numeric Targets)
        max_lin = self.profile.get('max_linear_score', 0)
        # Handle NaN case if correlation failed
        if pd.isna(max_lin): max_lin = 0.0
            
        if max_lin > 0.7:
             msg = "Strong linear relationship detected (>0.7 correlation). Linear Regression likely to perform well."
             self.suggestions['messages'].append(msg)
             self.suggestions['priority_models'].append('Linear Regression')
        elif max_lin < 0.2 and self.profile.get('n_samples', 0) > 100 and pd.api.types.is_numeric_dtype(y):
             # Weak linear signal -> Non-linear models
             msg = "Weak linear signal. Prioritizing Non-Linear models (Random Forest, Decision Tree)."
             self.suggestions['messages'].append(msg)
             self.suggestions['priority_models'].extend(['Random Forest', 'Decision Tree'])

        # Rule 5: Class Balance (Classification)
        if not pd.api.types.is_numeric_dtype(y) or (pd.api.types.is_integer_dtype(y) and y.nunique() < 20): 
             # Rough check for classification
             if not y.empty:
                 val_counts = y.value_counts(normalize=True)
                 if not val_counts.empty:
                     min_class = val_counts.min()
                     if min_class < 0.1: # Less than 10%
                          msg = f"Class imbalance detected (minority class: {min_class:.1%}). Ensure 'balanced' class weights."
                          self.suggestions['messages'].append(msg)
