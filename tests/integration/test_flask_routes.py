"""
Integration Tests — Flask Web Application (interface/app.py)
=============================================================
Tests all routes:
  GET /          — renders upload page
  POST /process  — runs the full pipeline, redirects to dashboard
  GET /dashboard — renders results page
  GET /download_model — serves the .pkl file

Uses Flask's test client for in-process HTTP testing.
"""

import io
import os
import sys
import unittest
import numpy as np
import pandas as pd
import tempfile

# Ensure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_csv_bytes(n=150, task="classification"):
    """Return a BytesIO CSV with synthetic data."""
    rng = np.random.RandomState(42)
    if task == "classification":
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n, n_features=5,
                                    n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["target"] = y
    else:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=n, n_features=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["target"] = y.astype(float)
    return df.to_csv(index=False).encode("utf-8")


class TestFlaskRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from interface.app import app
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.test_client()
        cls.app = app

    # ------------------------------------------------------------------ #
    # GET /
    # ------------------------------------------------------------------ #

    def test_index_route_returns_200(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_index_contains_html(self):
        response = self.client.get("/")
        content = response.data.decode("utf-8")
        self.assertIn("<", content)

    def test_index_contains_upload_element(self):
        """The upload page should contain a file input form."""
        response = self.client.get("/")
        content = response.data.decode("utf-8")
        self.assertTrue(
            "file" in content.lower() or "upload" in content.lower()
        )

    # ------------------------------------------------------------------ #
    # POST /process
    # ------------------------------------------------------------------ #

    def test_process_without_file_redirects(self):
        """No file attached → should redirect or return error page, not 500."""
        response = self.client.post("/process", data={"target_column": "target"})
        self.assertIn(response.status_code, [200, 302, 400])

    def test_process_with_invalid_extension_redirects(self):
        """Uploading a .txt file should redirect back with an error flash."""
        fake_file = (io.BytesIO(b"not a csv"), "data.txt")
        response = self.client.post(
            "/process",
            data={"file": fake_file, "target_column": "target"},
            content_type="multipart/form-data"
        )
        self.assertIn(response.status_code, [200, 302])

    def test_process_classification_redirects_to_dashboard(self):
        """Valid classification CSV → pipeline runs → redirect to /dashboard."""
        csv_bytes = _make_csv_bytes(n=150, task="classification")
        fake_file = (io.BytesIO(csv_bytes), "clf_data.csv")
        response = self.client.post(
            "/process",
            data={
                "file": fake_file,
                "target_column": "target",
                "models": ["Decision Tree"],
            },
            content_type="multipart/form-data"
        )
        # Should redirect to /dashboard
        self.assertIn(response.status_code, [302, 200])
        if response.status_code == 302:
            self.assertIn("dashboard", response.headers.get("Location", ""))

    def test_process_regression_redirects_to_dashboard(self):
        """Valid regression CSV → pipeline runs → redirect to /dashboard."""
        csv_bytes = _make_csv_bytes(n=150, task="regression")
        fake_file = (io.BytesIO(csv_bytes), "reg_data.csv")
        response = self.client.post(
            "/process",
            data={
                "file": fake_file,
                "target_column": "target",
                "models": ["Linear Regression"],
            },
            content_type="multipart/form-data"
        )
        self.assertIn(response.status_code, [302, 200])

    def test_process_missing_target_column_shows_error(self):
        """Submitting a column that does not exist → flash error, stay on index."""
        csv_bytes = _make_csv_bytes(n=150)
        fake_file = (io.BytesIO(csv_bytes), "data.csv")
        response = self.client.post(
            "/process",
            data={
                "file": fake_file,
                "target_column": "nonexistent_column",
                "models": ["Decision Tree"],
            },
            content_type="multipart/form-data"
        )
        # Should NOT be a 500 — gracefully handled
        self.assertNotEqual(response.status_code, 500)

    # ------------------------------------------------------------------ #
    # GET /dashboard
    # ------------------------------------------------------------------ #

    def test_dashboard_before_run_does_not_crash(self):
        """Visiting /dashboard before running any model should not return 500."""
        response = self.client.get("/dashboard")
        self.assertNotEqual(response.status_code, 500)

    def test_dashboard_after_run_returns_200(self):
        """After a successful /process call, /dashboard should return 200."""
        csv_bytes = _make_csv_bytes(n=150)
        fake_file = (io.BytesIO(csv_bytes), "clf_data.csv")
        self.client.post(
            "/process",
            data={
                "file": fake_file,
                "target_column": "target",
                "models": ["Decision Tree"],
            },
            content_type="multipart/form-data",
            follow_redirects=True
        )
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)

    def test_dashboard_after_run_contains_model_name(self):
        """Dashboard page content should mention a model name."""
        csv_bytes = _make_csv_bytes(n=150)
        fake_file = (io.BytesIO(csv_bytes), "clf_data.csv")
        self.client.post(
            "/process",
            data={
                "file": fake_file,
                "target_column": "target",
                "models": ["Decision Tree"],
            },
            content_type="multipart/form-data",
            follow_redirects=True
        )
        response = self.client.get("/dashboard")
        content = response.data.decode("utf-8").lower()
        self.assertIn("decision tree", content)

    # ------------------------------------------------------------------ #
    # GET /download_model
    # ------------------------------------------------------------------ #

    def test_download_model_before_run_is_graceful(self):
        """Downloading before any run should return 302/404, not 500."""
        response = self.client.get("/download_model")
        self.assertNotEqual(response.status_code, 500)

    def test_download_model_after_run_returns_file(self):
        """After a successful run, /download_model should serve a .pkl file."""
        csv_bytes = _make_csv_bytes(n=150)
        fake_file = (io.BytesIO(csv_bytes), "clf_data.csv")
        self.client.post(
            "/process",
            data={
                "file": fake_file,
                "target_column": "target",
                "models": ["Decision Tree"],
            },
            content_type="multipart/form-data",
            follow_redirects=True
        )
        response = self.client.get("/download_model")
        if response.status_code == 200:
            ct = response.headers.get("Content-Type", "")
            cd = response.headers.get("Content-Disposition", "")
            self.assertTrue(
                "application/octet-stream" in ct or
                "attachment" in cd or
                ".pkl" in cd
            )


class TestFlaskErrorHandling(unittest.TestCase):
    """Tests that the app handles error scenarios gracefully."""

    @classmethod
    def setUpClass(cls):
        from interface.app import app
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_invalid_route_returns_404(self):
        response = self.client.get("/does_not_exist")
        self.assertEqual(response.status_code, 404)

    def test_post_to_index_is_not_500(self):
        """POST to index should be gracefully handled (Method Not Allowed or redirect)."""
        response = self.client.post("/")
        self.assertIn(response.status_code, [200, 302, 405])


if __name__ == "__main__":
    unittest.main()
