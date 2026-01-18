from src.config.config import DataConfig, TrainingConfig, ModelConfig
from src.data.ingestion import DataIngestion
from src.features.transformation import DataTransformation
from src.models.trainer import ModelTrainer

class TrainPipeline:
    def run(self):
        data_config = DataConfig()
        training_config = TrainingConfig()
        model_config = ModelConfig()

        df = DataIngestion(data_config).ingest()
        X_train, X_test, y_train, y_test = DataTransformation(training_config).split(df)
        model = ModelTrainer(model_config).train(X_train, y_train)
        return model
