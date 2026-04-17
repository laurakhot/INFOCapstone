import joblib
import os
import json
import numpy as np


# -------------------------------------------------------
# Handler 1: model_fn
# Called once when the endpoint starts up.
# Loads the model from the path SageMaker provides.
# -------------------------------------------------------
def model_fn(model_dir):
    """Load model from the model_dir provided by SageMaker."""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    print(f"✅ Model loaded from {model_path}")
    return model


# -------------------------------------------------------
# Handler 2: predict_fn
# Called on every request. Receives the deserialized
# input from input_fn (or raw data if no input_fn) and
# the loaded model. Returns raw prediction output.
# -------------------------------------------------------
def predict_fn(input_data, model):
    """Run inference on the input data."""
    # input_data will be a numpy array from input_fn below
    predictions = model.predict(input_data)
    # probabilities = model.predict_proba(input_data)

    return {
        "predictions": predictions.tolist(),
        # "probabilities": probabilities.tolist()
    }


# -------------------------------------------------------
# Handler 3: output_fn
# Serializes the output from predict_fn into the
# response format the caller expects (JSON here).
# -------------------------------------------------------
def output_fn(prediction, accept):
    """Serialize the prediction output to JSON."""
    if accept == "application/json" or accept == "*/*":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}. Use application/json.")


# -------------------------------------------------------
# Bonus: input_fn (optional but recommended)
# Deserializes the raw request body before predict_fn.
# If omitted, SageMaker passes raw bytes to predict_fn.
# -------------------------------------------------------
def input_fn(request_body, content_type):
    """Deserialize the request body into a numpy array."""
    if content_type == "application/json":
        data = json.loads(request_body)
        # Accepts either {"instances": [[1,2,3], [4,5,6]]}
        # or just [[1,2,3], [4,5,6]]
        if isinstance(data, dict) and "instances" in data:
            return np.array(data["instances"])
        return np.array(data)
    raise ValueError(f"Unsupported content type: {content_type}. Use application/json.")