import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect
import joblib
import threading
import uuid
import time
from datetime import datetime

# Ensure core_recommender is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_recommender.execution import ModelExecutor
from core_recommender.knowledge_base import MODEL_KNOWLEDGE
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
# JOBS: { job_id: { status: 'running'|'completed'|'failed', progress: 0, message: '', results: {}, error: '', history: [] } }
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
            json.dump(jobs, f, default=str)
    except:
        pass

# Initialize JOBS
JOBS = load_jobs()

@app.route('/')
def index():
    """
    Renders the main landing page.
    """
    return render_template('index.html')

def background_training(job_id, filepath, target_column, selected_models=None):
    """
    Runs the comprehensive model training pipeline in a background thread.
    """
    global LAST_RESULTS, JOBS
    
    # PUSH APP CONTEXT (Fix for RuntimeError)
    ctx = app.app_context()
    ctx.push()
    
    def update_history(message, icon="fas fa-info-circle", type="info"):
        """Helper to append to job history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with JOBS_LOCK:
            JOBS[job_id]['history'].append({
                'timestamp': timestamp,
                'message': message,
                'icon': icon,
                'type': type
            })
            JOBS[job_id]['message'] = message
            save_jobs(JOBS)

    try:
        # Update status
        with JOBS_LOCK:
            JOBS[job_id]['message'] = "Initializing data processing..."
            JOBS[job_id]['progress'] = 5
            save_jobs(JOBS)
        
        update_history("Starting the journey. Loading your dataset...", "fas fa-file-import")

        # 1. Load Data
        df = pd.read_csv(filepath)
        
        # --- FIX: ROBUST COLUMN NAMES ---
        df.columns = df.columns.str.strip()
        target_column = target_column.strip()
        # -------------------------------
        
        update_history(f"Dataset loaded. Found {df.shape[0]} rows and {df.shape[1]} columns.", "fas fa-table")

        # 2. Generate Data Visualizations
        with JOBS_LOCK:
             JOBS[job_id]['progress'] = 10
             save_jobs(JOBS)
        
        update_history("Visualizing data patterns to understand relationships...", "fas fa-chart-pie")

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
        
        update_history("Visualizations generated. Now preparing to teach the models.", "fas fa-glasses")

        # 3. Define progress callback
        def progress_update(percent, message):
            # This callback comes from execution.py
            # We want to log significant steps to history, not just update the bar
            with JOBS_LOCK:
                if JOBS[job_id]['message'] != message:
                    JOBS[job_id]['progress'] = percent
                    JOBS[job_id]['message'] = message
                    save_jobs(JOBS)
            
            # Heuristic to determine if this is a "story" update vs a micro-update
            # If the message is new and distinctive, add to history
            # (We filter out repeat rapid updates in the frontend or here)
            # For now, let's just log "Training [Model]" type messages
            if "Training" in message or "Finished" in message or "Evaluating" in message:
                icon = "fas fa-cog fa-spin" if "Training" in message else "fas fa-check-circle"
                update_history(message, icon)

        # 4. Run Model Pipeline
        update_history("Initializing the Model Executor...", "fas fa-rocket")
        executor = ModelExecutor(task_type='auto', n_jobs=-1) 
        results = executor.run(
            df, 
            target_column=target_column, 
            progress_callback=progress_update,
            include_models=selected_models
        )
        
        update_history("All models have been trained and evaluated.", "fas fa-flag-checkered", "success")

        # 5. Generate Diagnostic Plots for Best Model
        update_history(f"The Champion is {results['best_model']['name']}! Generating final diagnostics...", "fas fa-trophy")
        
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
        # Inject Knowledge Base Description
        best_model_name = best_model['name']
        # Fuzzy match or direct match
        knowledge = MODEL_KNOWLEDGE.get(best_model_name, {})
        # If not exact match (e.g. "KNN (Classification)"), try to find partial
        if not knowledge:
            for k, v in MODEL_KNOWLEDGE.items():
                if k in best_model_name:
                    knowledge = v
                    break
        
        final_results = {
            'summary': {
                'task': results['task_type'].upper(),
                'best_model': best_model['name'],
                'primary_metric': list(best_model['metrics'].keys())[0] if best_model['metrics'] else 'Score', 
                'score': float(list(best_model['metrics'].values())[0] or 0.0) if best_model['metrics'] else 0.0,
                'description': knowledge # Inject rich description
            },
            'leaderboard': results['leaderboard'],
            'images': {
                'heatmap': 'static/images/heatmap.png',
                'histograms': 'static/images/histograms.png',
                'diagnostics_1': 'static/images/roc_curve.png' if results['task_type'] == 'classification' else 'static/images/residuals.png',
                'diagnostics_2': 'static/images/confusion_matrix.png' if results['task_type'] == 'classification' else 'static/images/pred_vs_actual.png',
                'importance': 'static/images/feature_importance.png' if 'importances' in importance_data else None,
            },
            'explainability': results['best_model']['explainability'],
            'model_knowledge': MODEL_KNOWLEDGE, # Pass full KB for comparison
            'best_model': results['best_model']
        }
        
        with JOBS_LOCK:
            LAST_RESULTS = final_results
            JOBS[job_id]['status'] = 'completed'
            JOBS[job_id]['progress'] = 100
            JOBS[job_id]['message'] = "Finished!"
            JOBS[job_id]['history'].append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'message': "Process Complete. Redirecting...",
                'icon': "fas fa-check-double",
                'type': "success"
            })
            save_jobs(JOBS)

    except Exception as e:
        import traceback
        traceback.print_exc()
        with JOBS_LOCK:
            JOBS[job_id]['status'] = 'failed'
            JOBS[job_id]['error'] = str(e)
            JOBS[job_id]['history'].append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'message': f"Error: {str(e)}",
                'icon': "fas fa-exclamation-triangle",
                'type': "danger"
            })
            save_jobs(JOBS)
    
    finally:
        # Clean up context
        if 'ctx' in locals():
            ctx.pop()

@app.route('/process', methods=['POST'])
def process():
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
            if ',' in m:
                selected_models.extend([sub.strip() for sub in m.split(',')])
            else:
                selected_models.append(m.strip())

    if not selected_models:
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
            'error': '',
            'history': [] # Initialize History
        }
        JOBS[job_id]['history'].append({
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'message': "Job submitted successfully.",
            'icon': "fas fa-play",
            'type': "info"
        })
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
        job = JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        response = {
            'status': job['status'],
            'progress': job['progress'],
            'message': job['message'],
            'history': job.get('history', []),
            'error': job['error']
        }
        
        if job['status'] == 'completed':
            response['redirect'] = url_for('dashboard')
            
        return jsonify(response)

@app.route('/dashboard')
def dashboard():
    if not LAST_RESULTS:
        return redirect(url_for('index'))
    return render_template('dashboard.html', results=LAST_RESULTS, knowledge=MODEL_KNOWLEDGE)

@app.route('/download_model')
def download_model():
    return send_from_directory(app.config['MODEL_FOLDER'], 'best_model.pkl', as_attachment=True)

if __name__ == '__main__':
    # Suppress Flask's werkzeug request logging (GET /status spam)
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    print("\\n * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)\\n")
    app.run(debug=True, port=5000)
