import bentoml
import numpy as np
from bentoml.io import JSON

# Import trained model
model_ref = bentoml.sklearn.import_model(
    "marks_predictor",
    model_uri="best_model.pkl"
)

# Create model runner
model_runner = model_ref.to_runner()

# Create BentoML service
svc = bentoml.Service(
    "marks_service",
    runners=[model_runner]
)

# Prediction API
@svc.api(input=JSON(), output=JSON())
async def predict(input_data):

    hours = np.array([[input_data["hours"]]])

    result = await model_runner.predict.async_run(hours)

    return {
        "predicted_marks": float(result[0])
    }
