import os
import uuid
import markdown
import sys

# Windows Fix: Set joblib temp folder to a stable local path to avoid FileNotFoundError during multiprocessing
JOBLIB_TEMP = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp', 'joblib'))
os.makedirs(JOBLIB_TEMP, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = JOBLIB_TEMP

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, flash, session
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

try:
    from interface.job_manager import job_manager
except ImportError:
    from job_manager import job_manager

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey_dev_only')  # Set SECRET_KEY env var in production

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['IMAGE_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'images')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'models')

# Ensure required directories exist at runtime
for folder in [app.config['UPLOAD_FOLDER'], app.config['IMAGE_FOLDER'], app.config['MODEL_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Per-request result store keyed by session UUID.
# Replaces the global LAST_RESULTS variable which caused race conditions
# when multiple users (or Gunicorn workers) hit /process concurrently.
RESULT_STORE: dict = {}
MAX_STORE_SIZE = 20   # evict oldest entries to cap memory usage

# Maximum allowed CSV upload size (50 MB)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# --- Routes ---

@app.route('/')
def home():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Validates uploaded CSV and launches the AutoML pipeline in a background thread."""
    
    # 1. Validate file upload
    if 'file' not in request.files:
        if request.headers.get('Accept') == 'application/json' or request.is_json:
            return jsonify({'error': 'No file part in the request'}), 400
        flash('Error: No file part in the request', 'danger')
        return redirect('/')
    
    file = request.files['file']
    if file.filename == '':
        if request.headers.get('Accept') == 'application/json' or request.is_json:
            return jsonify({'error': 'No file selected'}), 400
        flash('Error: No file selected', 'danger')
        return redirect('/')
    
    if not file.filename.endswith('.csv'):
        if request.headers.get('Accept') == 'application/json' or request.is_json:
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        flash('Error: Only CSV files are allowed', 'danger')
        return redirect('/')

    # File size guard — reject uploads over 50 MB before saving to disk.
    file.seek(0, 2)                   # seek to end
    file_size = file.tell()
    file.seek(0)                      # rewind
    if file_size > MAX_UPLOAD_BYTES:
        msg = f'Error: File too large ({file_size // (1024*1024)} MB). Maximum allowed size is 50 MB.'
        if request.headers.get('Accept') == 'application/json' or request.is_json:
            return jsonify({'error': msg}), 400
        flash(msg, 'danger')
        return redirect('/')
    
    # 2. Get target column
    target_column = request.form.get('target_column', '').strip()
    if not target_column:
        if request.headers.get('Accept') == 'application/json' or request.is_json:
            return jsonify({'error': 'Target column is required'}), 400
        flash('Error: Target column is required', 'danger')
        return redirect('/')
    
    # 3. Parse selected models
    selected_models = request.form.getlist('models')
    if selected_models:
        flattened = []
        for model in selected_models:
            if ',' in model:
                flattened.extend([m.strip() for m in model.split(',')])
            else:
                flattened.append(model.strip())
        selected_models = flattened
    else:
        if request.headers.get('Accept') == 'application/json' or request.is_json:
            return jsonify({'error': 'Please select at least one model to train.'}), 400
        flash('Error: Please select at least one model to train.', 'danger')
        return redirect('/')
    
    # 4. Save uploaded file to unique path to prevent concurrency conflicts
    upload_filename = f"dataset_{uuid.uuid4().hex[:10]}.csv"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
    file.save(filepath)
    
    # 5. Launch non-blocking background job
    job_id = job_manager.start_job(
        filepath=filepath,
        target_column=target_column,
        selected_models=selected_models,
        image_folder=app.config['IMAGE_FOLDER'],
        model_folder=app.config['MODEL_FOLDER']
    )
    
    session['result_id'] = job_id
    
    # Return immediately with HTTP 202 Accepted (eliminates 30s cloud timeout)
    return jsonify({
        'job_id': job_id,
        'status': 'started',
        'message': 'Pipeline executing in background thread.'
    }), 202


@app.route('/job_status/<job_id>')
def job_status(job_id: str):
    """Poll endpoint to check background AutoML progress and status."""
    job = job_manager.get_job(job_id)
    if not job:
        return jsonify({'status': 'not_found', 'error': 'Job not found'}), 404
    
    # If complete, keep session synced
    if job['status'] == 'completed':
        session['result_id'] = job['id']
        RESULT_STORE[job['id']] = job['results']

    return jsonify({
        'job_id': job['id'],
        'status': job['status'],
        'progress': job.get('progress', 0),
        'step': job.get('current_step', ''),
        'error': job.get('error')
    })


@app.route('/dashboard')
def dashboard():
    """Displays the results dashboard."""
    job_id = request.args.get('job_id') or session.get('result_id')
    if not job_id:
        flash('No analysis results found. Please submit a dataset first.', 'info')
        return redirect('/')

    # Check job_manager first, then RESULT_STORE
    results = None
    job = job_manager.get_job(job_id)
    if job and job.get('status') == 'completed' and job.get('results'):
        results = job['results']
    elif job_id in RESULT_STORE:
        results = RESULT_STORE[job_id]

    if not results:
        flash('Results not ready or job expired. Please try again.', 'warning')
        return redirect('/')
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
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
