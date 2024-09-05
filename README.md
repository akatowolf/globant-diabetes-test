# globant-diabetes-test
Globant prediction readmission

## Description
The goal of this analysis is to understand the dataset's structure, identify patterns, and uncover insights that will inform subsequent modeling efforts.
The ML Project will be a prediction for readmissions of diabetic patients using multiclass models.

## Installation
- **Prerequisites**: Python 3.11
- **Setup**: 
    1. git clone -b master https://github.com/akatowolf/globant-diabetes-test.git
    2. Go to the project and set up a new environment "python -m venv myvenv"
    3. Activate venv "myvenv\Scripts\activate"
    4. Install requirements.txt "pip install -r requirements.txt"
    5. Initialize mlflow server "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000"

## For training new models
For train new models run the script "main.py". This will execute the fine-tuning pipeline.
To try new models, parameters and metrics edit "training_pipeline.py"

## For testing predictions
There is a csv in data/prediction/pred.csv with dummie data to test the production model.
The model is retrieved from the MLFlow model registry and it is named as "diabetes_xgboost"
Run the script "predict.py". This will execute the predictions and will store them in a csv file in "data/prediction/leads.csv"
Other mlruns are stored locally to reduce github weight

# EDA
The Exploratory Data Analysis is a notebook and is stored in Notebooks folder
The Notebook contains the insights extracted from the dataset

## Business problem
The goal of the model is to predict the readmission of diabetic patients using a multiclass classification approach.

## Modeling
The registred model is an XGBoost Classifier, for performance details go to the mlflow artifact
The choosed metric is recall
Other Classifiers were used and the performance is stored locally (MLPClassifier, CatBoostClassifier, RandomForestClassifier)

## Experiments
MLFlow is used for tracking experiments, metrics stored are recall, accuracy, precision, cv recall. 
The model is stored as an artifact.

## Results
The performance of the chosen model is as follows:
* Training Recall: 0.62
* Testing Recall: 0.48
The model is exhibiting some overfitting, likely due to the oversampling technique used. 
There is potential for improvement in the model’s performance through further data treatment and refinement.


