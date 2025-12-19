import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import io
from typing import List, Optional
from sklearn.metrics import roc_curve, auc, confusion_matrix

# --- Configuration ---
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# =========================================================================
# 📊 Data Understanding (F9)
# =========================================================================

def plot_correlation_heatmap(df: pd.DataFrame, target_column: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a Seaborn heatmap showing the correlation between all features.
    
    Args:
        df: DataFrame containing all features and the target.
        target_column: The name of the column representing the target variable.
        save_path: Optional path to save the plot (e.g., 'correlation.png').
        
    Returns:
        The Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr()
    
    # Optional: Focus heatmap on target correlation for easier interpretation
    k = 15  # Number of variables for heatmap
    cols = corr.nlargest(k, target_column)[target_column].index
    cm = np.corrcoef(df[cols].values.T)
    
    sns.heatmap(cm, 
                annot=True, 
                square=True, 
                fmt='.2f', 
                ax=ax, 
                cmap='coolwarm',
                cbar_kws={'label': 'Correlation Coefficient'},
                yticklabels=cols.values.tolist(), 
                xticklabels=cols.values.tolist())
    
    ax.set_title(f"Feature Correlation Heatmap (Top {k} correlated with '{target_column}')", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        
    # How to return plot object for web display:
    # buffer = io.BytesIO()
    # fig.savefig(buffer, format='png')
    # buffer.seek(0)
    # return buffer.getvalue()
    
    return fig


def plot_feature_histograms(df: pd.DataFrame, features: List[str], save_path: Optional[str] = None) -> Figure:
    """
    Generates histograms for key feature distributions.
    
    Args:
        df: DataFrame containing the data.
        features: List of feature names to plot.
        save_path: Optional path to save the plot.
        
    Returns:
        The Matplotlib Figure object.
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
        fig.savefig(save_path)
        
    return fig

# =========================================================================
# 📈 Classification Diagnostics (F10)
# =========================================================================

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates the Receiver Operating Characteristic (ROC) curve.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_proba: Target scores, usually the probability of the positive class.
        model_name: Name of the model for the plot title/legend.
        save_path: Optional path to save the plot.
        
    Returns:
        The Matplotlib Figure object.
    """
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
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
        fig.savefig(save_path)
        
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a Seaborn Confusion Matrix heatmap.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: Array of class labels (e.g., ['No', 'Yes']).
        model_name: Name of the model for the plot title.
        save_path: Optional path to save the plot.
        
    Returns:
        The Matplotlib Figure object.
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
        fig.savefig(save_path)
        
    return fig


# =========================================================================
# 🧠 Model Interpretation (General)
# =========================================================================

def plot_feature_importance(feature_names: List[str], importances: np.ndarray, model_name: str, top_n: int = 15, save_path: Optional[str] = None) -> Figure:
    """
    Generates a horizontal bar chart showing the Feature Importance (for tree-based models).
    
    Args:
        feature_names: List of feature names.
        importances: Array of feature importance scores.
        model_name: Name of the model.
        top_n: Number of top features to display.
        save_path: Optional path to save the plot.
        
    Returns:
        The Matplotlib Figure object.
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
        fig.savefig(save_path)
        
    return fig

def plot_coefficient_bar_chart(feature_names: List[str], coefficients: np.ndarray, model_name: str, is_odds_ratio: bool = False, top_n: int = 15, save_path: Optional[str] = None) -> Figure:
    """
    Generates a bar chart of Coefficients/Weights (for Linear/Logistic Regression).
    
    Args:
        feature_names: List of feature names.
        coefficients: Array of coefficients (weights).
        model_name: Name of the model.
        is_odds_ratio: If True, plots exp(coefficients) for Logistic Regression interpretation.
        top_n: Number of top/bottom features to display.
        save_path: Optional path to save the plot.
        
    Returns:
        The Matplotlib Figure object.
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
    sns.barplot(x='Value', y='Feature', data=coef_df, ax=ax, palette='vlag') # vlag shows positive/negative clearly
    
    ax.set_title(f'Top {top_n} Feature Coefficients - {model_name}{title_suffix}', fontsize=14)
    ax.set_xlabel(y_label)
    ax.set_ylabel('Feature Name')
    plt.axvline(0, color='grey', linestyle='--') # Add line at zero for clear interpretation
    plt.tight_layout()
    
    if save_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path)
        
    return fig

# =========================================================================
# 📈 Regression Diagnostics
# =========================================================================

def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a scatter plot of Predicted vs. Actual values.
    
    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        model_name: Name of the model.
        save_path: Optional path to save the plot.
        
    Returns:
        The Matplotlib Figure object.
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
        fig.savefig(save_path)
        
    return fig

def plot_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a Residual Plot to check for homoscedasticity.
    
    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        model_name: Name of the model.
        save_path: Optional path to save the plot.
        
    Returns:
        The Matplotlib Figure object.
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
        fig.savefig(save_path)
        
    return fig

def plot_qq_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates a Q-Q Plot (Quantile-Quantile) to check normality of residuals.
    
    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        model_name: Name of the model.
        save_path: Optional path to save the plot.
        
    Returns:
        The Matplotlib Figure object.
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
        fig.savefig(save_path)
        
    return fig

def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str, save_path: Optional[str] = None) -> Figure:
    """
    Generates the Precision-Recall Curve.
    
    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities (expecting positive class probabilities or 2D array).
        model_name: Name of the model.
        save_path: Optional path to save the plot.
        
    Returns:
        The Matplotlib Figure object.
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
        fig.savefig(save_path)
        
    return fig