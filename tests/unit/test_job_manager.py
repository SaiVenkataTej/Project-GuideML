"""
Unit tests for JobManager and asynchronous background task tracking.
"""

import os
import time
import tempfile
import unittest
import pandas as pd
from interface.job_manager import JobManager
from interface.app import app


class TestJobManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.temp_dir, 'images')
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Create small test dataset
        self.csv_path = os.path.join(self.temp_dir, 'test.csv')
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            'x2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 5,
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5
        })
        df.to_csv(self.csv_path, index=False)

    def test_job_manager_lifecycle(self):
        """Test starting a job, polling progress, and reaching completed state."""
        manager = JobManager(max_stored_jobs=5, job_timeout_seconds=60)
        job_id = manager.start_job(
            filepath=self.csv_path,
            target_column='target',
            selected_models=['Logistic Regression'],
            image_folder=self.image_dir,
            model_folder=self.model_dir
        )
        self.assertIsNotNone(job_id)

        job = manager.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertIn(job['status'], ['running', 'completed'])

        # Wait up to 15 seconds for job to complete
        start_wait = time.time()
        while time.time() - start_wait < 15:
            job = manager.get_job(job_id)
            if job['status'] in ('completed', 'failed'):
                break
            time.sleep(0.3)

        final_job = manager.get_job(job_id)
        self.assertEqual(final_job['status'], 'completed')
        self.assertIsNotNone(final_job['results'])
        self.assertEqual(final_job['progress'], 100)

    def test_job_manager_not_found(self):
        """Test looking up an unknown job ID."""
        manager = JobManager()
        self.assertIsNone(manager.get_job('non-existent-id'))

    def test_flask_job_status_endpoint(self):
        """Test the /job_status/<id> Flask route."""
        with app.test_client() as client:
            # 404 on invalid ID
            res = client.get('/job_status/nonexistent-id')
            self.assertEqual(res.status_code, 404)


if __name__ == '__main__':
    unittest.main()
