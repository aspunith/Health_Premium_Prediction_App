import os
print(os.path.exists("C:/Users/punit/OneDrive/Desktop/AI ML/Machine Learning/Project_1_Healthcare_Premium_Prediction(Regression)/Health Premium prediction app/artifacts/model_rest.joblib"))  # Should print True if the file exists

import joblib


# Define absolute paths for local testing
absolute_model_rest_path = "C:/Users/punit/OneDrive/Desktop/AI ML/Machine Learning/Project_1_Healthcare_Premium_Prediction(Regression)/Health Premium prediction app/artifacts/model_rest.joblib"
absolute_model_young_path = "C:/Users/punit/OneDrive/Desktop/AI ML/Machine Learning/Project_1_Healthcare_Premium_Prediction(Regression)/Health Premium prediction app/artifacts/model_young.joblib"
absolute_scaler_rest_path = "C:/Users/punit/OneDrive/Desktop/AI ML/Machine Learning/Project_1_Healthcare_Premium_Prediction(Regression)/Health Premium prediction app/artifacts/scaler_rest.joblib"
absolute_scaler_young_path = "C:/Users/punit/OneDrive/Desktop/AI ML/Machine Learning/Project_1_Healthcare_Premium_Prediction(Regression)/Health Premium prediction app/artifacts/scaler_young.joblib"

# Define fallback relative paths
current_dir = os.path.dirname(__file__)
relative_model_rest_path = os.path.join(current_dir, "artifacts", "model_rest.joblib")
relative_model_young_path = os.path.join(current_dir, "artifacts", "model_young.joblib")
relative_model__scaler_young_path = os.path.join(current_dir, "artifacts", "scaler_young.joblib")
relative_model__scaler_rest_path = os.path.join(current_dir, "artifacts", "scaler_rest.joblib")

# Choose path based on availability
model_rest_path = absolute_model_rest_path if os.path.exists(absolute_model_rest_path) else relative_model_rest_path
model_young_path = absolute_model_young_path if os.path.exists(absolute_model_young_path) else relative_model_young_path
scaler_young_path = absolute_scaler_young_path if os.path.exists(absolute_scaler_young_path) else relative_model__scaler_young_path
scaler_rest_path = absolute_scaler_rest_path if os.path.exists(absolute_scaler_rest_path) else relative_model__scaler_rest_path

# Load the model
model_rest = joblib.load(model_rest_path)
model_young = joblib.load(model_young_path)
scaler_rest = joblib.load(scaler_rest_path)
scaler_young = joblib.load(scaler_young_path)

print("Current Directory:", os.path.dirname(__file__))
print("Model Rest Path:", model_rest_path)
