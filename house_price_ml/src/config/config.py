from dataclasses import dataclass

@dataclass
class DataConfig:
    raw_data_path: str = "data/raw/housing.csv"
    processed_data_path: str = "data/processed/processed.csv"

@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class ModelConfig:
    model_path: str = "artifacts/model.pkl"
