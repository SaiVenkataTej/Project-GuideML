"""
Job Manager Module
==================
Manages non-blocking background execution of the AutoML pipeline.
Eliminates HTTP 30-second gateway timeouts on cloud platforms (Render, Heroku)
and avoids browser hanging by running training jobs inside managed daemon threads.
"""

import os
import time
import uuid
import threading
import traceback
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from core_recommender.execution import ModelExecutor
from core_recommender.exceptions import GuideMLError, DataValidationError
from core_recommender.logger import get_logger

logger = get_logger(__name__)


class JobManager:
    """Thread-safe manager for background AutoML execution jobs."""

    def __init__(self, max_stored_jobs: int = 30, job_timeout_seconds: Optional[int] = None):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.max_stored_jobs = max_stored_jobs
        if job_timeout_seconds is not None:
            self.job_timeout_seconds = job_timeout_seconds
        else:
            self.job_timeout_seconds = int(os.environ.get('GUIDEML_JOB_TIMEOUT', 1800))

    def start_job(
        self,
        filepath: str,
        target_column: str,
        selected_models: Optional[List[str]],
        image_folder: str,
        model_folder: str
    ) -> str:
        """Creates and launches a new AutoML pipeline job in a background daemon thread.

        Args:
            filepath: Path to the saved dataset CSV.
            target_column: The target prediction column name.
            selected_models: List of model names to evaluate.
            image_folder: Directory to store diagnostic PNG plots.
            model_folder: Directory to store the exported best_model.pkl.

        Returns:
            str: Unique job_id (UUID4).
        """
        job_id = str(uuid.uuid4())

        with self._lock:
            # Evict oldest jobs if exceeding max_stored_jobs
            if len(self._jobs) >= self.max_stored_jobs:
                oldest_job_id = next(iter(self._jobs))
                del self._jobs[oldest_job_id]

            self._jobs[job_id] = {
                'id': job_id,
                'status': 'running',
                'progress': 5,
                'current_step': 'Initializing pipeline & loading data...',
                'results': None,
                'error': None,
                'start_time': time.time(),
                'end_time': None
            }

        worker = threading.Thread(
            target=self._run_pipeline_worker,
            args=(job_id, filepath, target_column, selected_models, image_folder, model_folder),
            daemon=True,
            name=f"guideml-job-{job_id[:8]}"
        )
        worker.start()
        logger.info(f"🚀 Started background job {job_id} for target '{target_column}'")
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the job record safely."""
        with self._lock:
            return self._jobs.get(job_id)

    def _update_job(self, job_id: str, **kwargs):
        """Thread-safe update of job properties."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)

    def _run_pipeline_worker(
        self,
        job_id: str,
        filepath: str,
        target_column: str,
        selected_models: Optional[List[str]],
        image_folder: str,
        model_folder: str
    ):
        """Worker function executing in the background thread."""
        start_time = time.time()

        def progress_callback(progress_pct: int, message: str):
            # Check for elapsed time timeout watchdog
            if (time.time() - start_time) > self.job_timeout_seconds:
                raise TimeoutError(
                    f"Job exceeded maximum processing limit of {self.job_timeout_seconds} seconds."
                )
            self._update_job(job_id, progress=progress_pct, current_step=message)

        try:
            # 1. Load and clean DataFrame
            self._update_job(job_id, progress=10, current_step="Ingesting dataset...")
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            target_column = target_column.strip()

            if target_column not in df.columns:
                raise DataValidationError(f"Target column '{target_column}' not found in dataset.", column=target_column)

            # 2. EDA visual generation
            self._update_job(job_id, progress=15, current_step="Generating feature correlation & histograms...")
            from core_recommender.visualization import (
                plot_correlation_heatmap,
                plot_feature_histograms,
                plot_roc_curve,
                plot_confusion_matrix,
                plot_predicted_vs_actual,
                plot_residual_plot,
                plot_feature_importance,
                plot_coefficient_bar_chart
            )

            # Clean old visual files
            for f in os.listdir(image_folder):
                if f.endswith('.png'):
                    try:
                        os.remove(os.path.join(image_folder, f))
                    except Exception:
                        pass

            heatmap_path = os.path.join(image_folder, 'heatmap.png')
            plot_correlation_heatmap(df, target_column, save_path=heatmap_path)

            hist_path = os.path.join(image_folder, 'histograms.png')
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            plot_feature_histograms(df, numeric_cols[:6], save_path=hist_path)

            # 3. ModelExecutor setup & run
            self._update_job(job_id, progress=20, current_step="Configuring model architectures...")
            executor = ModelExecutor(random_state=42)
            shap_path = os.path.join(image_folder, 'shap_summary.png')

            results = executor.run(
                df,
                target_column,
                include_models=selected_models,
                progress_callback=progress_callback,
                shap_output_path=shap_path
            )

            # 4. Generate diagnostics plots
            self._update_job(job_id, progress=92, current_step="Rendering model diagnostic charts...")
            task_type = results['task_type']
            best_model_data = results['best_model']
            diagnostics = best_model_data['diagnostics']

            if task_type == 'classification':
                n_classes = len(np.unique(diagnostics.get('y_true', []))) if 'y_true' in diagnostics else 0
                if 'y_proba' in diagnostics and diagnostics['y_proba'] is not None and n_classes == 2:
                    roc_path = os.path.join(image_folder, 'roc_curve.png')
                    plot_roc_curve(
                        np.asarray(diagnostics['y_true']),
                        np.asarray(diagnostics['y_proba']),
                        best_model_data['name'],
                        save_path=roc_path
                    )

                if 'y_pred' in diagnostics:
                    cm_path = os.path.join(image_folder, 'confusion_matrix.png')
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
            else:
                if 'y_pred' in diagnostics:
                    pred_path = os.path.join(image_folder, 'predicted_vs_actual.png')
                    plot_predicted_vs_actual(
                        np.asarray(diagnostics['y_true']),
                        np.asarray(diagnostics['y_pred']),
                        best_model_data['name'],
                        save_path=pred_path
                    )

                    residual_path = os.path.join(image_folder, 'residuals.png')
                    plot_residual_plot(
                        np.asarray(diagnostics['y_true']),
                        np.asarray(diagnostics['y_pred']),
                        best_model_data['name'],
                        save_path=residual_path
                    )

            # Feature importance / coefficient plots
            tailored = best_model_data.get('tailored_diagnostics', {})
            feature_names = best_model_data.get('feature_names', [])

            if 'feature_importances_mdi' in tailored or 'feature_importances' in tailored:
                importances = tailored.get('feature_importances_mdi') or tailored.get('feature_importances', [])
                if importances and len(feature_names) == len(importances):
                    imp_path = os.path.join(image_folder, 'feature_importance.png')
                    plot_feature_importance(
                        feature_names,
                        importances,
                        best_model_data['name'],
                        save_path=imp_path
                    )
            elif 'coefficients' in tailored:
                coefs = tailored['coefficients']
                if coefs and len(feature_names) == len(coefs):
                    coef_path = os.path.join(image_folder, 'coefficients.png')
                    plot_coefficient_bar_chart(
                        feature_names,
                        coefs,
                        best_model_data['name'],
                        save_path=coef_path
                    )

            # 5. Export best model artifact
            self._update_job(job_id, progress=98, current_step="Exporting best model artifact...")
            model_path = os.path.join(model_folder, 'best_model.pkl')
            if executor.best_model_instance:
                executor.best_model_instance.export(model_path)

            # 6. Complete job
            self._update_job(
                job_id,
                status='completed',
                progress=100,
                current_step='Pipeline execution complete.',
                results=results,
                end_time=time.time()
            )
            logger.info(f"✅ Job {job_id} completed successfully in {time.time() - start_time:.2f}s")

        except (GuideMLError, DataValidationError, TimeoutError) as e:
            logger.warning(f"⚠️ Job {job_id} failed with expected error: {e}")
            self._update_job(
                job_id,
                status='failed',
                error=str(e),
                current_step=f"Failed: {str(e)}",
                end_time=time.time()
            )
        except Exception as e:
            err_msg = traceback.format_exc()
            logger.error(f"❌ Job {job_id} encountered unhandled exception:\n{err_msg}")
            self._update_job(
                job_id,
                status='failed',
                error=f"Unexpected pipeline error: {str(e)}",
                current_step=f"Failed: {str(e)}",
                end_time=time.time()
            )
        finally:
            # Clean up uploaded raw CSV after run
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as clean_err:
                logger.debug(f"Could not remove temp upload {filepath}: {clean_err}")


# Global singleton instance
job_manager = JobManager()
