from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi import FastAPI, Header, HTTPException
import os
from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")

# === Load the trained model once ===
model_path = os.path.join(os.path.dirname(__file__), 'iris.mdl')
model = joblib.load(model_path)

# === FastAPI app metadata ===
app = FastAPI(
    title="Iris Predictor API",
    description="A FastAPI service that predicts the Iris species based on flower measurements.",
    version="1.0.0"
)

# === Request schema for the /hello endpoint ===
class NameRequest(BaseModel):
    """
    Schema for input data containing a user's name.
    """
    name: str

@app.post("/hello", summary="Greet the user", response_description="A greeting message")
def hello(data: NameRequest):
    """
    Returns a personalized greeting message.

    Args:
        data (NameRequest): A JSON body with a `name` field.

    Returns:
        dict: A JSON response with a greeting message.
    """
    return {"message": f"Hello {data.name}"}

# === Request schema for the /predict endpoint ===
class IrisFeatures(BaseModel):
    """
    Schema for input features required to predict the Iris species.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict", summary="Predict Iris species", response_description="Predicted Iris class")
async def predict(features: IrisFeatures, x_api_token: str = Header(...)):
    """
    Predicts the species of an Iris flower based on four numerical features.

    Args:
        data (IrisFeatures): JSON body with flower measurements.

    Returns:
        dict: A JSON response with the predicted Iris species.
    """
    if x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([features.dict()])

    # Make prediction
    prediction = model.predict(input_df)

    # Return the predicted class
    return {"predicted_species": prediction[0]}
