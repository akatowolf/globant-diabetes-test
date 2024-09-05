from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import json

class Preprocessor:
    def __init__(self, numerical_features, categorical_features, apply_scaling=True, apply_encoding=True):
        # The paremeters 'apply_scaling' and 'apply_encoding' control the preprocessing
        # This is to evaluate different data preprocessing
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.apply_scaling = apply_scaling
        self.apply_encoding = apply_encoding
        self.scaler = StandardScaler() if apply_scaling else None
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) if apply_encoding else None

    def fit(self, X):
        if self.apply_scaling and self.numerical_features:
            self.scaler.fit(X[self.numerical_features])
        
        if self.apply_encoding and self.categorical_features:
            self.encoder.fit(X[self.categorical_features])

    def transform(self, X):
        result = pd.DataFrame()
        
        if self.apply_scaling and self.numerical_features:
            scaled_data = self.scaler.transform(X[self.numerical_features])
            result = pd.DataFrame(scaled_data, columns=self.numerical_features, index=X.index)
        
        if self.apply_encoding and self.categorical_features:
            encoded_data = self.encoder.transform(X[self.categorical_features])
            encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(self.categorical_features), index=X.index)
            result = pd.concat([result, encoded_df], axis=1)
        
        with open('../data/processed/training_feature_names.json', 'r') as file:
            columns = json.load(file)
        features = columns['training_features']
        for col in features:
            if col not in result.columns:
                result[col] = 0
        result = result[features]
        return result

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
