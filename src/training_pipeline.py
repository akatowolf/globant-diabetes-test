import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from load_data import DataLoader
from feature_engineering import FeatureEngineering
from preprocessor import Preprocessor
from sklearn.ensemble import RandomForestClassifier
import re
import os
import numpy as np

tracking_uri = "sqlite:///../mlflow.db"
artifact_location = os.path.abspath("../mlruns")
mlflow.set_tracking_uri(tracking_uri)

# Models for experimenting
models = {
    'NeuralNetwork': MLPClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'CatBoost': CatBoostClassifier(),
    'RandomForest': RandomForestClassifier(),   
}

class TrainingPipeline:
    def __init__(self, data_path, experiment_name='Diabetes_Classification'):
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.data_loader = DataLoader(filepath=self.data_path)
        self.feature_engineer = FeatureEngineering()
        self.preprocessor = None

        mlflow.set_experiment(self.experiment_name)
        mlflow.get_experiment_by_name(self.experiment_name)

    def preprocess_data(self):
        # Load data
        df = self.data_loader.load_data()
        
        # Feature Engineering
        df_preprocessed = self.feature_engineer.create_features(df)
        
        # Separate features and target
        X = df_preprocessed.drop('readmitted', axis=1)
        y = df_preprocessed['readmitted']
        
        # Select numerical and categorical features for scaling and encoding
        numerical_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Initialize the Preprocessor
        self.preprocessor = Preprocessor(numerical_features, categorical_features, apply_saving=True)
        
        # Apply preprocessing
        X_preprocessed = self.preprocessor.fit_transform(X)
        X_preprocessed.columns = [
            re.sub(r'[^A-Za-z0-9_]+', '', str(col)) for col in X_preprocessed.columns
        ]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)
        
        sample = y_train[y_train==0].shape

        # ROSE to improve umbalanced
        sampling_strategy = {0:sample[0], 1:sample[0], 2:sample[0]}
        rose = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_res, y_res = rose.fit_resample(X_train, y_train)
        noise = np.random.normal(0, 0.01, X_res.shape)
        X_train_rose = X_res + noise
        
        # print new size
        for label in y_train.unique():
            print(f"Class {label} shape after ROSE:", X_train_rose[y_res == label].shape)
        mlflow.log_param("ROSE", str(sampling_strategy))
        return X_train_rose, X_val, y_res, y_val

    def train_models(self, X_train, X_val, y_train, y_val):
        if mlflow.active_run():
            mlflow.end_run()

        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                # Train the model
                model.fit(X_train, y_train)
                
                # Predict on training data
                y_train_pred = model.predict(X_train)
                train_recall = recall_score(y_train, y_train_pred, average='weighted')
                train_report = classification_report(y_train, y_train_pred)
                
                # Predict on validation data
                y_val_pred = model.predict(X_val)
                val_recall = recall_score(y_val, y_val_pred, average='weighted')
                val_report = classification_report(y_val, y_val_pred)
                val_confusion_matrix = confusion_matrix(y_val, y_val_pred)
                
                # Log metrics
                mlflow.log_metric("train_recall", train_recall)
                mlflow.log_metric("val_recall", val_recall)
                mlflow.log_text(train_report, "train_classification_report.txt")
                mlflow.log_text(val_report, "val_classification_report.txt")
                mlflow.log_text(str(val_confusion_matrix), "val_confusion_matrix.txt")
                mlflow.log_text("This train used ROSE", "This train used ROSE")
                
                # Log the model
                mlflow.sklearn.log_model(model, model_name)
                
                print(f"{model_name} - Training Recall: {train_recall:.4f}")
                print(f"{model_name} - Validation Recall: {val_recall:.4f}")
                print(f"{model_name} - Training Classification Report:\n{train_report}")
                print(f"{model_name} - Validation Classification Report:\n{val_report}")
                print(f"{model_name} - Validation Confusion Matrix:\n{val_confusion_matrix}")

    def run(self):
        X_train, X_val, y_train, y_val = self.preprocess_data()
        self.train_models(X_train, X_val, y_train, y_val)
