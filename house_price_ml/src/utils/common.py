import os
import pickle

class FileUtils:
    def save_object(path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def load_object(path):
        with open(path, "rb") as f:
            return pickle.load(f)
