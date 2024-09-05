import pandas as pd
import numpy as np
import utils
class FeatureEngineering:
    def __init__(self):
        pass
    
    def create_features(self, data):
        """
        Create new features from the existing data.
        """
        df = data.copy()

        # Drop columns
        df = df.drop(['encounter_id', 'patient_nbr', 'examide', 'citoglipton', 'weight', 'payer_code', 'medical_specialty'], axis=1)

        #df=df['admission_source_id'].map(admission_source_id_mapping)

        # Defining features categories
        medicaments = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
                       'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                       'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
        
        # Creating a new variable to count number of med changes
        df['med_changes'] = df[medicaments].apply(lambda row: sum(1 if value not in ['No', 'Steady'] else 0 for value in row), axis=1)

        # Mapping medicaments
        medication_mapping = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 2}
        for column in medicaments:
            df[column] = df[column].map(medication_mapping)

        # Mapping admission_type_id
        admission_type_id_mapping = {1: 'Emergency', 2: 'Urgent', 3: 'Elective', 4: 'Other_admi', 5: 'Other_admi', 6: 'Other_admi', 7: 'Other_admi', 8: 'Other_admi'}
        df['admission_type_id'] = df['admission_type_id'].map(admission_type_id_mapping)

        # Mapping diagnostics
        def categorize_code(code):
            try:
                code = float(code)
                if 250 <= code < 251:
                    return 'Diabetes'
                elif 401 <= code <= 405:
                    return 'Hypertension'
                elif 410 <= code <= 429:
                    return 'Heart'
                else:
                    return 'Others'
            except ValueError:
                return 'Others'
        for column in ['diag_1','diag_2','diag_3']:
            df[column] = df[column].apply(categorize_code)
        
        # Mapping target
        try:
            target_mapping = {'NO': 0, '<30': 1, '>30': 2}
            if 'readmitted' in df.columns:
                df['readmitted'] = df['readmitted'].map(target_mapping)
        except:
            pass

        # Mapping race
        df['race'] = df['race'].apply(
            lambda x: 'Caucasian' if x == 'Caucasian' else
                      'African American' if x == 'African American' else 
                      'Other')
        
        # Mapping change
        df['change'] = df['change'].apply(lambda x : 0 if x=='No' else 1)

        # Mapping diabetesMed
        df['diabetesMed'] = df['diabetesMed'].apply(lambda x : 0 if x=='No' else 1)

        # Mapping discharge_disposition_id
        def map_discharge_disposition(discharge_id):
            if discharge_id in [1, 6, 8]:
                return 'Sent Home'
            elif discharge_id in [2, 3, 4, 5, 9, 12, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30]:
                return 'Transferred'
            elif discharge_id in [11, 19, 20, 21, 13, 14]:
                return 'Expired'
            else:
                return 'Other'
        df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(map_discharge_disposition)

        def map_admission_source(admission_id):
            if admission_id in [4, 5, 6, 10, 18, 22, 25, 26]:
                return 'Hospital Transfer'
            elif admission_id in [1, 2, 3]:
                return 'Referral'
            elif admission_id in [7, 8, 9, 20, 21]:
                return 'Emergency'
            elif admission_id in [11, 12, 13, 14, 23, 24]:
                return 'Other'
            else:
                return 'Other'
        df['admission_source_id'] = df['admission_source_id'].apply(map_admission_source)

        # Map A1Cresult
        df['A1Cresult'] = df['A1Cresult'].fillna('None')
        mapping_a1c = {'Norm': 0, 'None': 1, '>7': 2, '>8': 3}
        df['A1Cresult'] = df['A1Cresult'].map(mapping_a1c)

        # Map A1Cresult
        df['max_glu_serum'] = df['max_glu_serum'].fillna('None')
        mapping_a1c = {'Norm': 0, 'None': 1, '>200': 2, '>300': 3}
        df['max_glu_serum'] = df['max_glu_serum'].map(mapping_a1c)

        # Total services
        df['total_services'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
        df = df.drop(['number_outpatient', 'number_emergency', 'number_inpatient'], axis=1)

        # Clip outliers
        cols_to_clip = ['num_medications', 'total_services', 'number_diagnoses']
        for column in cols_to_clip:
            df[column] = utils.clip_upper(df, column)

        return df
