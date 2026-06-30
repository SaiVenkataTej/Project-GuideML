import os
import tempfile

# Windows Fix: Set joblib temp folder to a stable local path to avoid FileNotFoundError and deadlocks during multiprocessing in tests
JOBLIB_TEMP = os.path.join(tempfile.gettempdir(), 'joblib_tests')
os.makedirs(JOBLIB_TEMP, exist_ok=True)
os.environ['JOBLIB_TEMP_FOLDER'] = JOBLIB_TEMP
