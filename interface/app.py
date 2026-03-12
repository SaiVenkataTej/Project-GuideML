import os
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
    print(f"DEBUG: Raw 'models' from form: {selected_models}") # Debug print
    
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
            if 'y_proba' in diagnostics and diagnostics['y_proba'] is not None:
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
        
        # Feature importance plots
        if 'feature_importance' in best_model_data:
            imp_data = best_model_data['feature_importance']
            if imp_data and imp_data.get('type') == 'tree':
                imp_path = os.path.join(app.config['IMAGE_FOLDER'], 'feature_importance.png')
                plot_feature_importance(
                    list(imp_data['features'].keys()),
                    list(imp_data['features'].values()),
                    best_model_data['name'],
                    save_path=imp_path
                )
            elif imp_data and imp_data.get('type') == 'linear':
                coef_path = os.path.join(app.config['IMAGE_FOLDER'], 'coefficients.png')
                plot_coefficient_bar_chart(
                    list(imp_data['features'].keys()),
                    list(imp_data['features'].values()),
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
    
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"ERROR in /process:\n{error_msg}")
        # Try to use project logger if possible
        try:
            from core_recommender.logger import get_logger
            logger = get_logger(__name__)
            logger.error(f"AutoML Pipeline failed: {str(e)}\n{error_msg}")
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
            'diagnostics_1': 'roc_curve.png' if task_type == 'classification' else 'residuals.png',
            'diagnostics_2': 'confusion_matrix.png' if task_type == 'classification' else 'predicted_vs_actual.png',
            'importance': 'feature_importance.png' if os.path.exists(os.path.join(app.config['IMAGE_FOLDER'], 'feature_importance.png')) else None,
            'shap': 'shap_summary.png' if os.path.exists(os.path.join(app.config['IMAGE_FOLDER'], 'shap_summary.png')) else None
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

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
