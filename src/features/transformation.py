import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.config import TrainingConfig

class DataTransformation:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def split(self, df: pd.DataFrame):
        X = df.drop("price", axis=1)
        y = df["price"]
        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
