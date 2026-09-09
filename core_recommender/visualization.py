import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import io
import os
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
from typing import List, Optional, Dict, Any
from sklearn.metrics import roc_curve, auc, confusion_matrix
from PIL import Image
from core_recommender.logger import get_logger

logger = get_logger(__name__)


def _save_and_resize(fig, save_path, target_size=(800, 600)):
    # Save with transparent background
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    try:
        img = Image.open(save_path).convert("RGBA")
        # Resize using BILINEAR while preserving aspect ratio
        img.thumbnail(target_size, resample=Image.Resampling.BILINEAR)
        # Create a blank transparent canvas
        new_img = Image.new("RGBA", target_size, (255, 255, 255, 0))
        # Center the resized image
        new_img.paste(img, ((target_size[0] - img.width) // 2, (target_size[1] - img.height) // 2), img)
        new_img.save(save_path, "PNG")
    except OSError as e:
        logger.error(f"Failed to resize image {save_path}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error resizing image {save_path}: {e}", exc_info=True)


# --- Configuration ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =========================================================================
# 📊 Data Understanding (F9)
# =========================================================================

def plot_correlation_heatmap(df: pd.DataFrame, target_column: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a Seaborn heatmap showing the correlation between all features.

    Rationale:
    ----------
    - **Multicollinearity Detection**: Highly correlated features (e.g., > 0.95) can destabilize linear models.
    - **Feature Selection**: Helps identify which features are strongly related to the target variable.

    Args:
        df: DataFrame containing all features and the target.
        target_column: The name of the column representing the target variable.
        save_path: Optional path to save the plot (e.g., 'correlation.png').

    Returns:
        Figure: The Matplotlib Figure object containing the heatmap.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Robustness Fix: If target_column is not numeric, temporarily encode it for the heatmap
    plot_df = df.copy()
    if target_column in df.columns and not pd.api.types.is_numeric_dtype(df[target_column]):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        plot_df[target_column] = le.fit_transform(df[target_column].astype(str))
        logger.debug(f"Encoded non-numeric target '{target_column}' for heatmap calculation.")

    corr = plot_df.corr(numeric_only=True)
    
    # Optional: Focus heatmap on target correlation for easier interpretation
    k = min(15, len(corr.columns)) # Ensure k is within bounds
    if target_column in corr.columns:
        cols = corr.nlargest(k, target_column)[target_column].index
        cm = np.corrcoef(plot_df[cols].values.T)
    else:
        # Fallback if target still not found or no numeric columns
        cols = corr.columns[:k]
        cm = corr.iloc[:k, :k].values
    
    sns.heatmap(cm, 
                annot=True, 
                square=True, 
                fmt='.2f', 
                ax=ax, 
                cmap='coolwarm',
                cbar_kws={'label': 'Correlation Coefficient'},
                yticklabels=cols.tolist(), 
                xticklabels=cols.tolist())
    
    ax.set_title(f"Feature Correlation Heatmap (Top {k} correlated with '{target_column}')", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        _save_and_resize(fig, save_path)
        
    # How to return plot object for web display:
    # buffer = io.BytesIO()
    # fig.savefig(buffer, format='png')
    # buffer.seek(0)
    # return buffer.getvalue()
    
    return fig


def plot_feature_histograms(df: pd.DataFrame, features: List[str], save_path: Optional[str] = None) -> Figure:
    """
    Generates histograms for key feature distributions to visualize data spread and skewness.

    Args:
        df: DataFrame containing the data.
        features: List of feature names to plot.
        save_path: Optional path to save the plot.

    Returns:
        Figure: The Matplotlib Figure object containing the histograms.
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols # Calculate required rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.histplot(data=df, x=feature, kde=True, ax=axes[i], color='teal')
            axes[i].set_title(f'Distribution of {feature}', fontsize=12)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle('Key Feature Distributions', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        _save_and_resize(fig, save_path)
        
    return fig

# =========================================================================
# 📈 Classification Diagnostics (F10)
# =========================================================================

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates the Receiver Operating Characteristic (ROC) curve to evaluate classifier performance.

    Rationale:
    ----------
    - **Trade-off Analysis**: Visualizes the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity).
    - **AUC**: The Area Under Curve provides a single scalar value to compare models; 0.5 is random guessing, 1.0 is perfect.

    Args:
        y_true: True binary labels (0 or 1).
        y_proba: Target scores, usually the probability of the positive class.
        model_name: Name of the model for the plot title/legend.
        save_path: Optional path to save the plot.

    Returns:
        Figure: The Matplotlib Figure object containing the ROC curve.
    """
    # Calculate ROC curve and AUC
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # ROBUSTNESS FIX: Handle 2D probability arrays (N samples, N classes)
    # roc_curve expects a 1D array of scores for the positive class.
    if y_proba.ndim == 2:
        if y_proba.shape[1] == 2:
            # Binary case: take the second column (usually class 1)
            y_score = y_proba[:, 1]
        else:
            # Multiclass case or something else: ravel or warn
            y_score = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
    else:
        y_score = y_proba

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'{model_name} ROC curve (AUC = {roc_auc:.4f})')
            
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.50)')
    
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f'Receiver Operating Characteristic (ROC) - {model_name}', fontsize=14)
    ax.legend(loc="lower right")
    
    if save_path:
        _save_and_resize(fig, save_path)
        
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a Seaborn Confusion Matrix heatmap to visualize misclassifications.

    Rationale:
    ----------
    - **Error Type Identification**: Shows exactly *how* the model is confused (e.g., False Positives vs False Negatives).
    - **Class Imbalance**: Reveals if the model is ignoring minority classes.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Array of class labels (e.g., ['No', 'Yes']) for axis annotation.
        model_name: Name of the model for the plot title.
        save_path: Optional path to save the plot.

    Returns:
        Figure: The Matplotlib Figure object containing the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_df, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                cbar=False,
                linewidths=.5,
                linecolor='black',
                ax=ax)
    
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        _save_and_resize(fig, save_path)
        
    return fig


# =========================================================================
# 🧠 Model Interpretation (General)
# =========================================================================

def plot_feature_importance(feature_names: List[str], importances: np.ndarray, model_name: str, top_n: int = 15, save_path: Optional[str] = None) -> Figure:
    """
    Generates a horizontal bar chart showing the Feature Importance (specifically for tree-based models).

    Args:
        feature_names: List of feature names.
        importances: Array of feature importance scores.
        model_name: Name of the model.
        top_n: Number of top features to display. Defaults to 15.
        save_path: Optional path to save the plot.

    Returns:
        Figure: The Matplotlib Figure object containing the feature importance plot.
    """
    # Combine, sort, and select top N features
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, color='darkgreen')
    
    ax.set_title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14)
    ax.set_xlabel('Feature Importance Score (e.g., Gini/Gain)')
    ax.set_ylabel('Feature Name')
    plt.tight_layout()
    
    if save_path:
        _save_and_resize(fig, save_path)
        
    return fig

def plot_coefficient_bar_chart(feature_names: List[str], coefficients: np.ndarray, model_name: str, is_odds_ratio: bool = False, top_n: int = 15, save_path: Optional[str] = None) -> Figure:
    """
    Generates a bar chart of Coefficients/Weights (for Linear/Logistic Regression).

    Args:
        feature_names: List of feature names.
        coefficients: Array of coefficients (weights).
        model_name: Name of the model.
        is_odds_ratio: If True, plots exp(coefficients) for Logistic Regression interpretation (Odds Ratio).
                       Defaults to False.
        top_n: Number of top feature coefficients (by absolute magnitude) to display. Defaults to 15.
        save_path: Optional path to save the plot.

    Returns:
        Figure: The Matplotlib Figure object containing the coefficient plot.
    """
    if is_odds_ratio:
        # For Logistic Regression, plot the Odds Ratio (exp(coef))
        plot_values = np.exp(coefficients)
        y_label = 'Odds Ratio (Exp(Coefficient))'
        title_suffix = ' (Odds Ratios)'
    else:
        plot_values = coefficients
        y_label = 'Coefficient Value'
        title_suffix = ' (Weights)'
        
    coef_df = pd.DataFrame({'Feature': feature_names, 'Value': plot_values})
    
    # Sort by absolute value, then select top/bottom features
    coef_df['Abs_Value'] = coef_df['Value'].abs()
    coef_df = coef_df.sort_values(by='Abs_Value', ascending=False).head(top_n)
    coef_df = coef_df.sort_values(by='Value', ascending=True) # Sort again for clean bar layout

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Value', y='Feature', data=coef_df, ax=ax, palette='vlag', hue='Feature', legend=False) # vlag shows positive/negative clearly
    
    ax.set_title(f'Top {top_n} Feature Coefficients - {model_name}{title_suffix}', fontsize=14)
    ax.set_xlabel(y_label)
    ax.set_ylabel('Feature Name')
    plt.axvline(0, color='grey', linestyle='--') # Add line at zero for clear interpretation
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        _save_and_resize(fig, save_path)
        
    return fig

def plot_shap_summary(model: Any, X: pd.DataFrame, model_name: str, save_path: str):
    """
    Generates a SHAP summary plot (bar) to show global feature impact.
    
    Args:
        model: The fitted model (typically the 'best_estimator' from a search or pipeline).
        X: The preprocessed feature matrix (DataFrame) for which to compute SHAP values.
        model_name: Descriptive name of the model for the plot title.
        save_path: Filesystem path to save the generated PNG.
    """
    plt.figure(figsize=(10, 6))
    
    if not HAS_SHAP:
        plt.text(0.5, 0.5, "SHAP Unavailable: shap library not installed", ha='center', va='center')
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        return

    try:
        # Performance & Memory Guard: Sample data for speed and low RAM footprint.
        # SHAP calculation can be computationally expensive (especially Kernel/Permutation).
        if len(X) > 50:
            X_sample = X.sample(50, random_state=42)
        else:
            X_sample = X
            
        # Explainer Dispatch: Extract the inner model if buried in a scikit-learn Pipeline.
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
             actual_model = model.named_steps['model']
        else:
             actual_model = model

        # Initialize SHAP Explainer - handles trees, linear models, and kernels automatically.
        explainer = shap.Explainer(actual_model, X_sample)
        shap_values = explainer(X_sample, check_additivity=False)
        
        # Render Global Feature Importance (Bar Plot Type).
        # max_display is capped at 10 to maintain dashboard readability.
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=10)
        
        plt.title(f"SHAP Global Impact Analysis - {model_name}", fontsize=14, pad=20)
        
        # Ensure target directory exists before saving (Windows robustness).
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    except Exception as e:
        logger.error(f"SHAP generation failed for {model_name}: {e}", exc_info=True)
        # Graceful failure: Render an error message on the plot artifact if SHAP fails.
        plt.text(0.5, 0.5, f"SHAP Unavailable: {str(e)}", ha='center', va='center')
        try:
            plt.savefig(save_path)
        except OSError as save_err:
            logger.error(f"Failed to save SHAP fallback image: {save_err}")
    finally:
        plt.close()

# =========================================================================
# 📈 Regression Diagnostics
# =========================================================================

def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a scatter plot of Predicted vs. Actual values to assess regression performance.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        model_name: Name of the model.
        save_path: Optional path to save the plot.

    Returns:
        Figure: The Matplotlib Figure object containing the scatter plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.6, color='teal', edgecolor='k')
    
    # Ideal line (y=x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction (Perfect Fit)')
    
    ax.set_title(f'Predicted vs. Actual - {model_name}', fontsize=14)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        _save_and_resize(fig, save_path)
        
    return fig

def plot_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a Residual Plot to check for homoscedasticity (constant variance of errors).

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        model_name: Name of the model.
        save_path: Optional path to save the plot.

    Returns:
        Figure: The Matplotlib Figure object containing the residual plot.
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.6, color='purple', edgecolor='w')
    ax.axhline(0, color='red', linestyle='--', lw=2)
    
    ax.set_title(f'Residual Plot - {model_name}', fontsize=14)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    plt.tight_layout()
    
    if save_path:
        _save_and_resize(fig, save_path)
        
    return fig

def plot_qq_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a Q-Q Plot (Quantile-Quantile) to visually check the normality of residuals.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        model_name: Name of the model.
        save_path: Optional path to save the plot.

    Returns:
        Figure: The Matplotlib Figure object containing the Q-Q plot.
    """
    import scipy.stats as stats
    
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    stats.probplot(residuals, dist="norm", plot=ax)
    
    ax.get_lines()[0].set_color('steelblue') # probplot points
    ax.get_lines()[0].set_markersize(5.0)
    ax.get_lines()[1].set_color('red')       # probplot line
    ax.get_lines()[1].set_linewidth(2.0)
    
    ax.set_title(f'Q-Q Plot of Residuals - {model_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        _save_and_resize(fig, save_path)
        
    return fig

def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates the Precision-Recall Curve, which is useful for imbalanced datasets.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities (usually for the positive class). 
                 Can be a 1D array of probabilities or a 2D array where the second column is the positive class.
        model_name: Name of the model.
        save_path: Optional path to save the plot.

    Returns:
        Figure: The Matplotlib Figure object containing the Precision-Recall curve.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # Handle inputs
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    if y_proba.ndim == 2:
        if y_proba.shape[1] >= 2:
             # Take positive class probability (index 1) for binary plot
             # For multi-class, this simple plot isn't sufficient without one-vs-rest loop, 
             # focusing on binary as per standard requirement or taking index 1.
             y_score = y_proba[:, 1]
        else:
             y_score = y_proba.ravel()
    else:
         y_score = y_proba

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, color='purple', lw=2, label=f'AP (Average Precision) = {avg_precision:.4f}')
    
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc="lower left")
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        _save_and_resize(fig, save_path)
        
    return fig