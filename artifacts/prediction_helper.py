import pandas as pd
import os
import joblib

# Define the base directory dynamically (for local or deployed environment)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the artifacts directory based on the current environment
# Local environment: Assuming the `artifacts` folder is in the same directory as this script
artifacts_dir = os.path.join(current_dir, "artifacts")

# For deployed environment on Streamlit, if running from a mount or specific path, adjust accordingly
if "mount" in current_dir:  # For Streamlit deployment (change as needed)
    artifacts_dir = "/mount/src/health_premium_prediction_app/artifacts"
else:  # Local development environment
    artifacts_dir = os.path.abspath(os.path.join(current_dir, "artifacts"))

# Debugging: Print the artifacts directory to ensure it's correct
print("Artifacts Directory:", artifacts_dir)

# Paths for model and scaler files
model_rest_path = os.path.join(artifacts_dir, "model_rest.joblib")
model_young_path = os.path.join(artifacts_dir, "model_young.joblib")
scaler_rest_path = os.path.join(artifacts_dir, "scaler_rest.joblib")
scaler_young_path = os.path.join(artifacts_dir, "scaler_young.joblib")

# Debugging: Print paths to ensure correctness
print("Model Rest Path:", model_rest_path)
print("Model Young Path:", model_young_path)
print("Scaler Rest Path:", scaler_rest_path)
print("Scaler Young Path:", scaler_young_path)

# Verify that all necessary files exist
required_files = [model_rest_path, model_young_path, scaler_rest_path, scaler_young_path]
for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

# Load models and scalers
model_rest = joblib.load(model_rest_path)
model_young = joblib.load(model_young_path)
scaler_rest = joblib.load(scaler_rest_path)
scaler_young = joblib.load(scaler_young_path)

print("Models and scalers loaded successfully!")



def calculate_normalised_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)
    max_score = 14  # Update this if new conditions are added
    min_score = 0
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)
    return normalized_risk_score


def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalised_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # Map inputs to DataFrame columns
    for key, value in input_dict.items():
        if key == 'Gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key == 'Region':
            if value == 'Northwest':
                df['region_Northwest'] = 1
            elif value == 'Southeast':
                df['region_Southeast'] = 1
            elif value == 'Southwest':
                df['region_Southwest'] = 1
        elif key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_Unmarried'] = 1
        elif key == 'BMI Category':
            if value == 'Obesity':
                df['bmi_category_Obesity'] = 1
            elif value == 'Overweight':
                df['bmi_category_Overweight'] = 1
            elif value == 'Underweight':
                df['bmi_category_Underweight'] = 1
        elif key == 'Smoking Status':
            if value == 'Occasional':
                df['smoking_status_Occasional'] = 1
            elif value == 'Regular':
                df['smoking_status_Regular'] = 1
        elif key == 'Employment Status':
            if value == 'Salaried':
                df['employment_status_Salaried'] = 1
            elif value == 'Self-Employed':
                df['employment_status_Self-Employed'] = 1
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == "Genetical Risk":
            df['genetical_risk'] = value

    # Compute normalized risk score
    df['normalised_risk_score'] = calculate_normalised_risk(input_dict['Medical History'])
    df = handle_scaling(input_dict['Age'], df)

    return df


def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def predict(input_dict):
    input_df = preprocess_input(input_dict)
    if input_dict['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)
    return int(prediction[0])
