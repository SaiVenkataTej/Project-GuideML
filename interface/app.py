import os
import markdown
import sys

# Windows Fix: Set joblib temp folder to a stable local path to avoid FileNotFoundError during multiprocessing
JOBLIB_TEMP = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp', 'joblib'))
os.makedirs(JOBLIB_TEMP, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = JOBLIB_TEMP

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, flash
import joblib
from typing import Dict, Any, List, Optional

# Add core_recommender package to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_recommender.execution import ModelExecutor
from core_recommender.knowledge_base import MODEL_KNOWLEDGE
from core_recommender.exceptions import DataValidationError, GuideMLError
from core_recommender.visualization import (
    plot_correlation_heatmap, plot_feature_histograms, plot_roc_curve,
    plot_confusion_matrix, plot_feature_importance, plot_coefficient_bar_chart,
    plot_predicted_vs_actual, plot_residual_plot, plot_shap_summary
)

app = Flask(__name__)
app.secret_key = 'supersecretkey_dev_only' # Required for flash messages

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['IMAGE_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'images')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'models')
 
# Global to store the latest results for the dashboard
LAST_RESULTS = None

# --- Routes ---

@app.route('/')
def home():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Processes the uploaded CSV and runs the AutoML pipeline synchronously."""
    global LAST_RESULTS
    
    # 1. Validate file upload
    if 'file' not in request.files:
        flash('Error: No file part in the request', 'danger')
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        flash('Error: No file selected', 'danger')
        return redirect('/')
    
    if not file.filename.endswith('.csv'):
        flash('Error: Only CSV files are allowed', 'danger')
        return redirect('/')
    
    # 2. Get target column
    target_column = request.form.get('target_column', '').strip()
    if not target_column:
        flash('Error: Target column is required', 'danger')
        return redirect('/')
    
    # 3. Parse selected models
    selected_models = request.form.getlist('models')
    
    if selected_models:
        # Flatten any comma-separated values
        flattened = []
        for model in selected_models:
            if ',' in model:
                flattened.extend([m.strip() for m in model.split(',')])
            else:
                flattened.append(model.strip())
        selected_models = flattened
    else:
        flash('Error: Please select at least one model to train.', 'danger')
        return redirect('/')
    
    # 4. Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
    file.save(filepath)
    
    try:
        # 5. Load data
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()  # Clean column names
        target_column = target_column.strip()
        
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in dataset'}), 400
        
        # 6. Generate data visualizations
        # Clean old images
        for f in os.listdir(app.config['IMAGE_FOLDER']):
            if f.endswith('.png'):
                file_path = os.path.join(app.config['IMAGE_FOLDER'], f)
                try:
                    # Windows robustness: check if exists and try to delete
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"DEBUG: Could not remove {f}: {e}")
                    pass
        
        heatmap_path = os.path.join(app.config['IMAGE_FOLDER'], 'heatmap.png')
        plot_correlation_heatmap(df, target_column, save_path=heatmap_path)
        
        hist_path = os.path.join(app.config['IMAGE_FOLDER'], 'histograms.png')
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        plot_feature_histograms(df, numeric_cols[:6], save_path=hist_path)
        
        # 7. Run ModelExecutor
        executor = ModelExecutor(random_state=42)
        results = executor.run(df, target_column, include_models=selected_models)
        
        # 8. Generate diagnostic plots
        task_type = results['task_type']
        # 8. Diagnostics Data
        best_model_data = results['best_model']
        diagnostics = best_model_data['diagnostics']
        
        # Note: Safety aliasing removed; all models now standardized to 'y_true'
        
        # 9. Plotting logic
        if task_type == 'classification':
            # ROC curve — only valid for binary classification
            n_classes = len(np.unique(diagnostics.get('y_true', []))) if 'y_true' in diagnostics else 0
            if 'y_proba' in diagnostics and diagnostics['y_proba'] is not None and n_classes == 2:
                roc_path = os.path.join(app.config['IMAGE_FOLDER'], 'roc_curve.png')
                plot_roc_curve(
                    np.asarray(diagnostics['y_true']),
                    np.asarray(diagnostics['y_proba']),
                    best_model_data['name'],
                    save_path=roc_path
                )
            
            if 'y_pred' in diagnostics:
                cm_path = os.path.join(app.config['IMAGE_FOLDER'], 'confusion_matrix.png')
                
                # Get classes from results or derive from y_true
                classes = results.get('classes')
                if classes is None:
                    classes = np.unique(diagnostics['y_true']).astype(str)
                    
                plot_confusion_matrix(
                    np.asarray(diagnostics['y_true']),
                    np.asarray(diagnostics['y_pred']),
                    np.asarray(classes),
                    best_model_data['name'],
                    save_path=cm_path
                )
        
        # Regression plots
        else:
            if 'y_pred' in diagnostics:
                pred_path = os.path.join(app.config['IMAGE_FOLDER'], 'predicted_vs_actual.png')
                plot_predicted_vs_actual(
                    np.asarray(diagnostics['y_true']),
                    np.asarray(diagnostics['y_pred']),
                    best_model_data['name'],
                    save_path=pred_path
                )
                
                residual_path = os.path.join(app.config['IMAGE_FOLDER'], 'residuals.png')
                plot_residual_plot(
                    np.asarray(diagnostics['y_true']),
                    np.asarray(diagnostics['y_pred']),
                    best_model_data['name'],
                    save_path=residual_path
                )
        
        # Feature importance / coefficient plots
        # Driven by tailored_diagnostics — each model exposes only what it has.
        tailored = best_model_data.get('tailored_diagnostics', {})
        feature_names = best_model_data.get('feature_names', [])

        # Tree-based models (Random Forest, Decision Tree)
        if 'feature_importances_mdi' in tailored or 'feature_importances' in tailored:
            importances = tailored.get('feature_importances_mdi') or tailored.get('feature_importances', [])
            if importances and len(feature_names) == len(importances):
                imp_path = os.path.join(app.config['IMAGE_FOLDER'], 'feature_importance.png')
                plot_feature_importance(
                    feature_names,
                    importances,
                    best_model_data['name'],
                    save_path=imp_path
                )

        # Linear models (Logistic Regression, Linear Regression, Linear SVM)
        elif 'coefficients' in tailored:
            coefs = tailored['coefficients']
            if coefs and len(feature_names) == len(coefs):
                coef_path = os.path.join(app.config['IMAGE_FOLDER'], 'coefficients.png')
                plot_coefficient_bar_chart(
                    feature_names,
                    coefs,
                    best_model_data['name'],
                    save_path=coef_path
                )
        
        # 9. Save best model
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model.pkl')
        if executor.best_model_instance:
            executor.best_model_instance.export(model_path)
        
        # 10. Store results globally
        LAST_RESULTS = results
        
        # 11. Redirect to dashboard
        return redirect('/dashboard')
    
    except DataValidationError as e:
        # Data problems are user-fixable — warn instead of error
        flash(f'Data issue: {e}', 'warning')
        return redirect('/')
    except GuideMLError as e:
        # All other typed pipeline errors
        flash(f'Pipeline error [{type(e).__name__}]: {e}', 'danger')
        return redirect('/')
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"ERROR in /process:\n{error_msg}")
        try:
            from core_recommender.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"Unhandled exception in AutoML Pipeline: {str(e)}\n{error_msg}")
        except:
            pass
        flash(f'An unexpected error occurred: {str(e)}', 'danger')
        return redirect('/')

@app.route('/dashboard')
def dashboard():
    """Displays the results dashboard."""
    if not LAST_RESULTS:
        return redirect('/')
    
    results = LAST_RESULTS
    
    # Prepare leaderboard
    leaderboard = results.get('leaderboard', [])
    
    # Determine primary metric
    task_type = results['task_type']
    if task_type == 'classification':
        primary_metric_key = 'F1 Score'
    else:
        primary_metric_key = 'RMSE'
    
    # Get best score
    best_score = results['best_model']['metrics'].get(primary_metric_key, 0.0)
    
    # Get knowledge base entry
    best_model_name = results['best_model']['name']
    
    # Simple mapping for knowledge base (stripping suffixes)
    kb_key = best_model_name
    for key in MODEL_KNOWLEDGE.keys():
        if key.lower() in best_model_name.lower():
            kb_key = key
            break
            
    knowledge = MODEL_KNOWLEDGE.get(kb_key, {})
    
    # Construct formatted results object for the dashboard template.
    # This architecture decoupling ensures the frontend only receives 
    # the necessary visual data and high-level summaries.
    img_folder = app.config['IMAGE_FOLDER']
    formatted_results = {
        'summary': {
            'best_model': best_model_name,
            'primary_metric': primary_metric_key,
            'score': best_score,
            'task': task_type.upper(),
            'description': {
                'story': knowledge.get('story', 'Model trained successfully.'),
                'best_for': knowledge.get('best_for', 'General analysis')
            }
        },
        'leaderboard': results.get('leaderboard', []),
        'images': {
            # Only set a filename if the file was actually generated on disk.
            # Prevents broken <img> tags when a plot is skipped (e.g. ROC for multiclass).
            'diagnostics_1': (
                'roc_curve.png'
                if task_type == 'classification' and os.path.exists(os.path.join(img_folder, 'roc_curve.png'))
                else 'residuals.png'
                if task_type == 'regression' and os.path.exists(os.path.join(img_folder, 'residuals.png'))
                else None
            ),
            'diagnostics_2': (
                'confusion_matrix.png'
                if task_type == 'classification' and os.path.exists(os.path.join(img_folder, 'confusion_matrix.png'))
                else 'predicted_vs_actual.png'
                if task_type == 'regression' and os.path.exists(os.path.join(img_folder, 'predicted_vs_actual.png'))
                else None
            ),
            # Show feature_importance.png for tree models, coefficients.png for linear models
            'importance': (
                'feature_importance.png'
                if os.path.exists(os.path.join(img_folder, 'feature_importance.png'))
                else 'coefficients.png'
                if os.path.exists(os.path.join(img_folder, 'coefficients.png'))
                else None
            ),
            'shap': 'shap_summary.png' if os.path.exists(os.path.join(img_folder, 'shap_summary.png')) else None
        },
        'explainability': {
            'parameters': results['best_model'].get('explainability', {}).get('parameters', {})
        },
        'best_model': {
            'pipeline_log': results['best_model'].get('pipeline_log', [])
        }
    }
    
    return render_template('dashboard.html', results=formatted_results)

@app.route('/download_model')
def download_model():
    """Downloads the trained model."""
    model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model.pkl')
    if os.path.exists(model_path):
        return send_from_directory(app.config['MODEL_FOLDER'], 'best_model.pkl', as_attachment=True)
    return jsonify({'error': 'Model not found'}), 404

@app.route('/docs')
def docs():
    """Renders the docs.md file as an HTML page."""
    docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'docs.md'))
    with open(docs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    html_content = markdown.markdown(content, extensions=['fenced_code', 'tables'])
    return render_template('docs.html', content=html_content)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
