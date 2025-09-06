from flask import Flask, request, jsonify, render_template
from model import COVIDModel
import joblib
import os

app = Flask(__name__)
model = COVIDModel()

# Flag to track if model is loaded
model_loaded = False

# Load the model before the first request
@app.before_request
def load_model_if_needed():
    global model_loaded
    if not model_loaded:
        try:
            model.load_model('covid_model.pkl')
            print("Successfully loaded the trained logistic regression model")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            
            fallback_model = LogisticRegression(random_state=42)
            X_dummy = np.random.rand(100, 18) 
            y_dummy = np.random.randint(0, 2, 100) 
            fallback_model.fit(X_dummy, y_dummy)
            
            model.model = fallback_model
            print("Using fallback model - this is for demonstration only!")
        
        model_loaded = True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.form.to_dict()
        
      
        input_data = {
            'Sex': data.get('sex', 'UNKNOWN'),
            'Chest_pain': data.get('chest_pain', 'NO'),
            'Cough': data.get('cough', 'NO'),
            'Diarrhea': data.get('diarrhea', 'NO'),
            'Fatigue_or_general_weakness': data.get('fatigue', 'NO'),
            'Fever': data.get('fever', 'NO'),
            'Headache': data.get('headache', 'NO'),
            'Thorax_(sore_throat)': data.get('sore_throat', 'NO'),
            'Nausea': data.get('nausea', 'NO'),
            'Runny_nose': data.get('runny_nose', 'NO'),
            'Vomiting': data.get('vomiting', 'NO'),
            'Loss_of_Taste': data.get('loss_of_taste', 'NO'),
            'Loss_of_Smell': data.get('loss_of_smell', 'NO')
        }
        
       
        result = model.predict(input_data)
        
    
        result_class = "positive" if result["prediction"] == "POSITIVE" else "negative"
        
        return render_template('index.html', 
                             prediction_text=f'COVID-19 Test Result: {result["prediction"]}',
                             confidence_text=f'Confidence: {result["confidence"]}',
                             result_class=result_class,
                             result=result)
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f'Error: {str(e)}',
                             result_class="error")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        # Make prediction
        result = model.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)