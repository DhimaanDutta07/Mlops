import mlflow.sklearn
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from src.utils.common import FileUtils
from src.config.config import ModelConfig
import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(
    repo_owner="DhimaanDutta07",
    repo_name="sec",
    mlflow=True
)

class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        mlflow.set_experiment("Mlops project")
        mlflow.sklearn.autolog()
        with mlflow.start_run():
            self.model.fit(X_train, y_train)
            FileUtils.save_object(self.config.model_path, self.model)
            return self.model
