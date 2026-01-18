from sklearn.linear_model import LinearRegression
from src.utils.common import FileUtils
from src.config.config import ModelConfig

class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        FileUtils.save_object(self.config.model_path, self.model)
        return self.model
