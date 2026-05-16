import pandas as pd
from pycaret.regression import *
import mlflow

# Load dataset
data = pd.read_csv("student.csv")

# Create MLflow experiment
mlflow.set_experiment("Student_Marks_Prediction")

with mlflow.start_run():

    # Setup AutoML
    reg = setup(
        data=data,
        target='marks',
        session_id=123,
        verbose=False
    )

    # Compare ML models
    best_model = compare_models()

    # Save best model
    save_model(best_model, 'best_model')

    # Log parameter
    mlflow.log_param("Model", str(best_model))

print("Training Completed")
