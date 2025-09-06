import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

class COVIDModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def load_model(self, model_path):
        """Load the trained model and extract feature names"""
        try:
            # Load the model
            self.model = joblib.load(model_path)
            print(f"Model loaded successfully. Model type: {type(self.model)}")
            
            # Try to extract feature names from the model
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                print(f"Feature names from model: {self.feature_names}")
            else:
                print("Model doesn't have feature names stored")
               
                self.feature_names = [
                    'Sex', 'Chest_pain', 'Cough', 'Diarrhea', 'Fatigue_or_general_weakness',
                    'Fever', 'Headache', 'Thorax_(sore_throat)', 'Nausea', 'Runny_nose',
                    'Vomiting', 'Loss_of_Taste', 'Loss_of_Smell', 'Symptom_Count',
                    'GI_Symptom_Count', 'GI_Symptom_Flag', 'Neuro_Symptom_Count', 'Neuro_Symptom_Flag'
                ]
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
        
    def preprocess_input(self, input_data):
        """Preprocess the input data to match training format"""
       
        df = pd.DataFrame([input_data])
        
        
        symptom_cols = ['Chest_pain', 'Cough', 'Diarrhea', 'Fatigue_or_general_weakness', 
                       'Fever', 'Headache', 'Thorax_(sore_throat)', 'Nausea', 'Runny_nose', 
                       'Vomiting', 'Loss_of_Taste', 'Loss_of_Smell']
        
        # Convert YES/NO to 1/0 for symptom count calculation
        symptom_values = df[symptom_cols].replace({'YES': 1, 'NO': 0, 'UNKNOWN': 0})
        df['Symptom_Count'] = symptom_values.sum(axis=1)
        
        # GI symptoms
        gi_cols = ["Diarrhea", "Vomiting", "Nausea"]
        gi_values = df[gi_cols].replace({'YES': 1, 'NO': 0, 'UNKNOWN': 0})
        df['GI_Symptom_Count'] = gi_values.sum(axis=1)
        df['GI_Symptom_Flag'] = (df['GI_Symptom_Count'] > 0).astype(int)
        
        # Neurological symptoms
        neuro_cols = ["Headache", "Loss_of_Smell", "Loss_of_Taste"]
        neuro_values = df[neuro_cols].replace({'YES': 1, 'NO': 0, 'UNKNOWN': 0})
        df['Neuro_Symptom_Count'] = neuro_values.sum(axis=1)
        df['Neuro_Symptom_Flag'] = (df['Neuro_Symptom_Count'] > 0).astype(int)
        
        # Encode categorical variables 
        categorical_cols = ['Sex', 'Chest_pain', 'Cough', 'Diarrhea', 'Fatigue_or_general_weakness',
                           'Fever', 'Headache', 'Thorax_(sore_throat)', 'Nausea', 'Runny_nose',
                           'Vomiting', 'Loss_of_Taste', 'Loss_of_Smell']
        
        for col in categorical_cols:
            if col in df.columns:
                # Based on your notebook, you replaced 'YES'->1, 'NO'->0, 'MALE'->1, 'FEMALE'->0
                df[col] = df[col].map({'YES': 1, 'NO': 0, 'MALE': 1, 'FEMALE': 0, 'UNKNOWN': 0})
                # Fill any NaN values with 0 (as your imputer used most_frequent which was likely 'NO')
                df[col] = df[col].fillna(0)
        
       
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0  
            
            
            X = df[self.feature_names]
        else:
            
            X = df
        
        
        print(f"Final features for prediction: {list(X.columns)}")
        print(f"Feature values: {X.values}")
        
        return X
    
    def predict(self, input_data):
        """Make a prediction on the input data"""
        try:
            processed_data = self.preprocess_input(input_data)
            
            prediction = self.model.predict(processed_data)
            
            # Check if the model has predict_proba method
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(processed_data)
                confidence = max(probability[0]) * 100
                probability_positive = probability[0][1] 
            else:
                confidence = 100 if prediction[0] == 1 else 0
                probability_positive = 1.0 if prediction[0] == 1 else 0.0
            
            return {
                'prediction': 'POSITIVE' if prediction[0] == 1 else 'NEGATIVE',
                'confidence': f"{confidence:.2f}%",
                'probability_positive': probability_positive
            }
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {
                'prediction': 'ERROR',
                'confidence': '0%',
                'error': str(e)
            }