import os
from datetime import datetime

class ExperimentContext:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(*args, **kwargs)
        return cls._instance

    def __init__(self, base_dir='experiments/'):
        self.timestamp = datetime.now().strftime('%Y_%m_%d_%H%M')
        self.run_dir = os.path.join(base_dir, f"model_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

    def get_path(self, name=None):
        return os.path.join(self.run_dir, name)