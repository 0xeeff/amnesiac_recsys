import os

import joblib


class State:
    def __init__(self, file_path, schema):
        self.file_path = file_path
        self.schema = schema
        self.is_loaded_from_checkpoint = False
        self.values = self.initialize()

    def initialize(self):
        if os.path.exists(self.file_path):
            self.is_loaded_from_checkpoint = True
            return joblib.load(self.file_path)
        else:
            # this must be the first time the app is run
            dir = os.path.dirname(self.file_path)
            os.makedirs(dir, exist_ok=True)
            # print(f"creating checkpoint {self.file_path}")
            joblib.dump(self.schema, self.file_path)
            return self.schema

    def update(self, new_state):
        joblib.dump(new_state, self.file_path)
        self.values = new_state
