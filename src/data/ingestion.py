import pandas as pd
from src.config.config import DataConfig

class DataIngestion:
    def __init__(self, config: DataConfig):
        self.config = config

    def ingest(self):
        df = pd.read_csv(self.config.raw_data_path)
        df.to_csv(self.config.processed_data_path, index=False)
        return df
