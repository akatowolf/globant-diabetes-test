from fine_tuning_xgboost import FineTuningPipeline
from training_pipeline import TrainingPipeline

data_path = '../data/raw/diabetic_data.csv'

# Uncomment to run training pipeline (the pipeline tries different models and store the runs in mlflow)
#pipeline = TrainingPipeline(data_path=data_path)

# The selected model for finetuning is XGBoost, this command allow to try the finetuning 
pipeline = FineTuningPipeline(data_path=data_path)
pipeline.run()
