import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import joblib
import threading
import uuid
import time

# Ensure core_recommender is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_recommender.execution import ModelExecutor
from core_recommender.visualization import (
    plot_correlation_heatmap, 
    plot_feature_histograms,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_predicted_vs_actual,
    plot_residual_plot
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['IMAGE_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'images')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'models')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Global store for demo purposes (In prod, use database/session)
# JOBS: { job_id: { status: 'running'|'completed'|'failed', progress: 0, message: '', results: {}, error: '' } }
JOBS_FILE = os.path.join(app.config['UPLOAD_FOLDER'], '.jobs.json')
LAST_RESULTS = {}
JOBS_LOCK = threading.Lock()

def load_jobs():
    if not os.path.exists(JOBS_FILE):
        return {}
    try:
        import json
        with open(JOBS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_jobs(jobs):
    try:
        import json
        with open(JOBS_FILE, 'w') as f:
            json.dump(jobs, f)
    except:
        pass

# Initialize JOBS
JOBS = load_jobs()

@app.route('/')
def index():
    """
    Renders the main landing page.
    
    Rationale:
    ----------
    - Serves as the entry point for the application.
    - Loads the single-page application (SPA) interface.
    """
    return render_template('index.html')

def background_training(job_id, filepath, target_column, selected_models=None):
    """
    Runs the comprehensive model training pipeline in a background thread.
    
    Rationale:
    ----------
    - **Non-blocking UX**: Training can take minutes to hours; blocking the HTTP request would timeout the browser.
    - **State Management**: Updates a global `JOBS` dictionary so the frontend can poll for progress.
    - **Atomic Updates**: Uses `threading.Lock` to strictly serialize write access to the shared `JOBS` state.
    
    Args:
        job_id: Unique identifier for the job.
        filepath: Absolute path to the uploaded CSV file.
        selected_models: List of model names to include (if filtering).
        target_column: The name of the target variable to predict.
    """
    global LAST_RESULTS, JOBS
    
    # PUSH APP CONTEXT (Fix for RuntimeError)
    # Using explicit push() avoids indenting the entire function
    ctx = app.app_context()
    ctx.push()
    
    try:
        # Update status
        with JOBS_LOCK:
            JOBS[job_id]['message'] = "Loading and cleaning data..."
            JOBS[job_id]['progress'] = 5
            save_jobs(JOBS)

        # 1. Load Data
        df = pd.read_csv(filepath)
        
        # --- FIX: ROBUST COLUMN NAMES ---
        # Strip leading/trailing whitespace from all column names to prevent KeyErrors
        df.columns = df.columns.str.strip()
        target_column = target_column.strip()
        # -------------------------------

        # 2. Generate Data Visualizations
        with JOBS_LOCK:
             JOBS[job_id]['message'] = "Generating initial visualizations..."
             JOBS[job_id]['progress'] = 10
             save_jobs(JOBS)

        # Clean old images
        for f in os.listdir(app.config['IMAGE_FOLDER']):
             if f.endswith('.png'):
                try: os.remove(os.path.join(app.config['IMAGE_FOLDER'], f))
                except: pass
            
        heatmap_path = os.path.join(app.config['IMAGE_FOLDER'], 'heatmap.png')
        plot_correlation_heatmap(df, target_column, save_path=heatmap_path)
        
        hist_path = os.path.join(app.config['IMAGE_FOLDER'], 'histograms.png')
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_column in numeric_cols: numeric_cols.remove(target_column)
        plot_feature_histograms(df, numeric_cols[:6], save_path=hist_path)

        # 3. Define progress callback
        def progress_update(percent, message):
            with JOBS_LOCK:
                if JOBS[job_id]['message'] != message: # Only update/save if message changed
                    JOBS[job_id]['progress'] = percent
                    JOBS[job_id]['message'] = message
                    save_jobs(JOBS)

        # 4. Run Model Pipeline
        executor = ModelExecutor(task_type='auto', n_jobs=-1) 
        results = executor.run(
            df, 
            target_column=target_column, 
            progress_callback=progress_update,
            include_models=selected_models
        )
        
        # 5. Generate Diagnostic Plots for Best Model
        with JOBS_LOCK:
            JOBS[job_id]['message'] = "Generating diagnostic plots..."
            save_jobs(JOBS)
        
        best_model = results['best_model']
        diag_data = best_model['diagnostics']
        y_test = diag_data['y_test']
        y_pred = diag_data['y_pred']
        name = best_model['name']
        
        if results['task_type'] == 'classification':
            if 'y_proba' in diag_data:
                y_proba = diag_data['y_proba']
                roc_path = os.path.join(app.config['IMAGE_FOLDER'], 'roc_curve.png')
                try:
                    plot_roc_curve(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba, name, save_path=roc_path)
                except Exception as e:
                    print(f"ROC Plot failed: {e}")
            
            cm_path = os.path.join(app.config['IMAGE_FOLDER'], 'confusion_matrix.png')
            classes = np.unique(y_test)
            plot_confusion_matrix(y_test, y_pred, classes, name, save_path=cm_path)
            
        else:
            res_path = os.path.join(app.config['IMAGE_FOLDER'], 'residuals.png')
            plot_residual_plot(y_test, y_pred, name, save_path=res_path)
            
            pred_path = os.path.join(app.config['IMAGE_FOLDER'], 'pred_vs_actual.png')
            plot_predicted_vs_actual(y_test, y_pred, name, save_path=pred_path)

        # 5.5 Generate Feature Importance Plot
        importance_data = results['best_model']['explainability']['importance']
        if 'importances' in importance_data and 'feature_names' in importance_data:
            imp_path = os.path.join(app.config['IMAGE_FOLDER'], 'feature_importance.png')
            # For Linear models, importance is coefficients
            if results['best_model']['name'].startswith('Linear'):
                from core_recommender.visualization import plot_coefficient_bar_chart
                plot_coefficient_bar_chart(
                    importance_data['feature_names'], 
                    np.array(importance_data['importances']), 
                    results['best_model']['name'],
                    save_path=imp_path
                )
            else:
                from core_recommender.visualization import plot_feature_importance
                plot_feature_importance(
                    importance_data['feature_names'], 
                    np.array(importance_data['importances']), 
                    results['best_model']['name'],
                    save_path=imp_path
                )

        # 6. Save Model Artifact
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model.pkl')
        joblib.dump(executor.best_model_instance, model_path)

        # 7. Prepare response
        final_results = {
            'summary': {
                'task': results['task_type'].upper(),
                'best_model': best_model['name'],
                'primary_metric': list(best_model['metrics'].keys())[0] if best_model['metrics'] else 'Score', 
                'score': float(list(best_model['metrics'].values())[0] or 0.0) if best_model['metrics'] else 0.0
            },
            'leaderboard': results['leaderboard'],
            'images': {
                'heatmap': 'static/images/heatmap.png',
                'histograms': 'static/images/histograms.png',
                'diagnostics_1': 'static/images/roc_curve.png' if results['task_type'] == 'classification' else 'static/images/residuals.png',
                'diagnostics_2': 'static/images/confusion_matrix.png' if results['task_type'] == 'classification' else 'static/images/pred_vs_actual.png',
                'importance': 'static/images/feature_importance.png' if 'importances' in importance_data else None,
            },
            'explainability': results['best_model']['explainability']
        }
        
        with JOBS_LOCK:
            LAST_RESULTS = final_results
            JOBS[job_id]['status'] = 'completed'
            JOBS[job_id]['progress'] = 100
            JOBS[job_id]['message'] = "Finished!"
            save_jobs(JOBS)

    except Exception as e:
        import traceback
        traceback.print_exc()
        with JOBS_LOCK:
            JOBS[job_id]['status'] = 'failed'
            JOBS[job_id]['error'] = str(e)
            save_jobs(JOBS)
    
    finally:
        # Clean up context
        if 'ctx' in locals():
            ctx.pop()

@app.route('/process', methods=['POST'])
def process():
    """
    Handles file upload and initiates the training job.
    
    Rationale:
    ----------
    - **Async Handoff**: Validates inputs immediately but delegates heavy processing to a background thread.
    - **Robust Validation**: Checks file type (CSV) and existence of target column before starting.
    
    Returns:
        JSON response with `job_id` for polling.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    target_column = request.form.get('target_column')
    if not target_column:
        return jsonify({'error': 'Target column must be specified'}), 400

    # Extract selected models (list)
    raw_models = request.form.getlist('models[]') 
    selected_models = []
    if raw_models:
        for m in raw_models:
            # Handle comma-separated values (e.g., "Linear Regression, Logistic Regression")
            if ',' in m:
                selected_models.extend([sub.strip() for sub in m.split(',')])
            else:
                selected_models.append(m.strip())

    if not selected_models:
        # Fallback to all if none selected
        selected_models = None

    # Create Job ID
    job_id = str(uuid.uuid4())
    
    # Save File
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}.csv')
    file.save(filepath)
    
    # Initialize Job State
    with JOBS_LOCK:
        JOBS[job_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Starting...',
            'error': ''
        }
        save_jobs(JOBS)

    # Start Background Thread
    thread = threading.Thread(target=background_training, args=(job_id, filepath, target_column, selected_models))
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'queued', 'job_id': job_id})

@app.route('/status/<job_id>')
def job_status(job_id):
    global JOBS
    with JOBS_LOCK:
        # Reload to handle potential restart losses (though in-memory usually fine if not restarted)
        # But if restart happened, load_jobs() at startup will have it.
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        response = {
            'status': job['status'],
            'progress': job['progress'],
            'message': job['message'],
            'error': job['error']
        }
        
        if job['status'] == 'completed':
            response['redirect'] = url_for('dashboard')
            
        return jsonify(response)

@app.route('/dashboard')
def dashboard():
    if not LAST_RESULTS:
        return redirect(url_for('index'))
    return render_template('dashboard.html', results=LAST_RESULTS)

@app.route('/download_model')
def download_model():
    return send_from_directory(app.config['MODEL_FOLDER'], 'best_model.pkl', as_attachment=True)

if __name__ == '__main__':
    # Suppress Flask's werkzeug request logging (GET /status spam)
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # Only show errors, not every request
    
    print("\n * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)\n")
    app.run(debug=True, port=5000)
