import mlflow
import mlflow.xgboost
import pandas as pd
from load_data import DataLoader
from feature_engineering import FeatureEngineering
from preprocessor import Preprocessor
import re


# Set the tracking URI
tracking_uri = "sqlite:///../mlflow.db"
mlflow.set_tracking_uri(tracking_uri)

# Model name
model_name = "diabetes_xgboost"
model_version = "1"

# Load model from model registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.xgboost.load_model(model_uri)

class PredictionPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_loader = DataLoader(filepath=self.data_path)
        self.feature_engineer = FeatureEngineering()
        self.preprocessor = None

    def preprocess_data(self):
        # Load data
        df = self.data_loader.load_data()

        # Save ID's
        ids = df['patient_nbr']

        # Feature Engineering
        df_preprocessed = self.feature_engineer.create_features(df)
        
        # Select numerical and categorical features for scaling and encoding
        numerical_features = df_preprocessed.select_dtypes(include=['number']).columns.tolist()
        categorical_features = df_preprocessed.select_dtypes(include=['object']).columns.tolist()
        
        # Initialize the Preprocessor
        self.preprocessor = Preprocessor(numerical_features, categorical_features)
        
        # Apply preprocessing
        X_preprocessed = self.preprocessor.fit_transform(df_preprocessed)
        X_preprocessed.columns = [
            re.sub(r'[^A-Za-z0-9_]+', '', str(col)) for col in X_preprocessed.columns
        ]
        print(X_preprocessed.columns)
        return ids, X_preprocessed

    def predict(self):
        ids, X_preprocessed = self.preprocess_data()
        
        # Make predictions
        predictions = model.predict(X_preprocessed)
        
        # Combine IDs with predictions
        results = pd.DataFrame({
            'id': ids,
            'prediction': predictions
        })
        
        return results

if __name__ == "__main__":
    data_path = "../data/prediction/pred.csv"
    save_preds = "../data/prediction/leads.csv"
    pipeline = PredictionPipeline(data_path)
    predictions = pipeline.predict()
    target_mapping = {0: 'NO', 1:'<30', 2:'>30'}
    predictions['prediction'] = predictions['prediction'].map(target_mapping)
    predictions.to_csv(save_preds, index=False)
    print("Predictions saved")
