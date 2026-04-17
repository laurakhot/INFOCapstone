# Export the model
import joblib
import os

os.makedirs("model", exist_ok=True)
joblib.dump(your_model_variable, "model/model.joblib")