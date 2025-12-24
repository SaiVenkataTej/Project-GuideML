import os
import sys
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import joblib

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
# Storing the last run results to render in dashboard
LAST_RESULTS = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global LAST_RESULTS
    
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

    try:
        # 1. Save File
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
        file.save(filepath)
        
        # 2. Load Data
        df = pd.read_csv(filepath)
        
        if target_column not in df.columns:
             return jsonify({'error': f"Column '{target_column}' not found in CSV"}), 400

        # 3. Generate Data Visualizations (Heatmap & Hists)
        # Clean old images
        for f in os.listdir(app.config['IMAGE_FOLDER']):
            os.remove(os.path.join(app.config['IMAGE_FOLDER'], f))
            
        heatmap_path = os.path.join(app.config['IMAGE_FOLDER'], 'heatmap.png')
        plot_correlation_heatmap(df, target_column, save_path=heatmap_path)
        
        hist_path = os.path.join(app.config['IMAGE_FOLDER'], 'histograms.png')
        # Plot top 6 numeric features
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_column in numeric_cols: numeric_cols.remove(target_column)
        plot_feature_histograms(df, numeric_cols[:6], save_path=hist_path)

        # 4. Run Model Pipeline
        executor = ModelExecutor(task_type='auto', n_jobs=-1) 
        results = executor.run(df, target_column=target_column)
        
        # 5. Generate Diagnostic Plots for Best Model
        best_model = results['best_model']
        diag_data = best_model['diagnostics']
        y_test = diag_data['y_test']
        y_pred = diag_data['y_pred']
        name = best_model['name']
        
        if results['task_type'] == 'classification':
            # ROC Curve
            if 'y_proba' in diag_data:
                # Handle binary vs multi-class shape for plotting
                y_proba = diag_data['y_proba']
                roc_path = os.path.join(app.config['IMAGE_FOLDER'], 'roc_curve.png')
                try:
                    plot_roc_curve(y_test, y_proba[:, 1] if y_proba.ndim > 1 else y_proba, name, save_path=roc_path)
                except Exception as e:
                    print(f"ROC Plot failed: {e}")
            
            # Confusion Matrix
            cm_path = os.path.join(app.config['IMAGE_FOLDER'], 'confusion_matrix.png')
            classes = np.unique(y_test)
            plot_confusion_matrix(y_test, y_pred, classes, name, save_path=cm_path)
            
        else:
            # Regression Plots
            res_path = os.path.join(app.config['IMAGE_FOLDER'], 'residuals.png')
            plot_residual_plot(y_test, y_pred, name, save_path=res_path)
            
            pred_path = os.path.join(app.config['IMAGE_FOLDER'], 'pred_vs_actual.png')
            plot_predicted_vs_actual(y_test, y_pred, name, save_path=pred_path)

        # 6. Save Model Artifact
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'best_model.pkl')
        joblib.dump(executor.best_model_instance, model_path)

        # 7. Prepare response
        LAST_RESULTS = {
            'summary': {
                'task': results['task_type'].upper(),
                'best_model': best_model['name'],
                'primary_metric': list(best_model['metrics'].keys())[0], 
                'score': list(best_model['metrics'].values())[0]
            },
            'leaderboard': results['leaderboard'],
            'images': {
                'heatmap': 'static/images/heatmap.png',
                'histograms': 'static/images/histograms.png',
                'diagnostics_1': 'static/images/roc_curve.png' if results['task_type'] == 'classification' else 'static/images/residuals.png',
                'diagnostics_2': 'static/images/confusion_matrix.png' if results['task_type'] == 'classification' else 'static/images/pred_vs_actual.png',
            }
        }
        
        return jsonify({'status': 'success', 'redirect': url_for('dashboard')})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    if not LAST_RESULTS:
        return redirect(url_for('index'))
    return render_template('dashboard.html', results=LAST_RESULTS)

@app.route('/download_model')
def download_model():
    return send_from_directory(app.config['MODEL_FOLDER'], 'best_model.pkl', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
