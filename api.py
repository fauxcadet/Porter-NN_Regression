import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import os 
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError # Needed for custom_objects
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# --- 0. Configuration and Artifacts ---

# Define custom objects for model loading (CRITICAL FIX)
CUSTOM_OBJECTS = {
    'mse': tf.keras.metrics.mean_squared_error,
    'mae': tf.keras.metrics.mean_absolute_error
}

# Replace placeholders with your actual data derived during training
# These values MUST be derived from your final x.columns list
FINAL_COLUMNS = [
    'market_id', 'subtotal', 'total_items', 'num_distinct_items', 
    'min_item_price', 'max_item_price', 'total_onshift_partners', 
    'total_busy_partners', 'total_outstanding_orders', 'store_avg_delivery_time',
    'hours', 'day', 'is_lunch_rush', 'is_dinner_rush', 'is_weekend',
    'category_american', 'category_mexican', 'category_thai', 
    # NOTE: Add ALL 14+ OHE category columns here.
]
# Replace with the actual global mean time calculated from your training data
GLOBAL_MEAN_TIME = 45.0 


# --- 1. Load Artifacts ---

# We wrap the loading in a check to allow the API to start even if files are missing 
# (useful for testing, but in production, files must exist)
try:
    final_model = tf.keras.models.load_model('delivery_time_model.h5', custom_objects=CUSTOM_OBJECTS)
    fitted_scaler = joblib.load('fitted_scaler.joblib')
    print("Model and Scaler loaded successfully.")
    
except Exception as e:
    # If loading fails, print a warning and set placeholders to None
    print(f"Warning: Model artifact loading failed. {e}")
    final_model = None
    fitted_scaler = None

# --- 2. CORE PREDICTION PIPELINE FUNCTION ---

def predict_delivery_time(raw_data: dict, model, scaler, feature_columns) -> float:
    """
    Takes raw order data, processes it through the feature engineering and scaling pipeline,
    and returns the predicted delivery time.
    """
    if model is None or scaler is None:
        raise ValueError("Model or Scaler not loaded. Cannot run prediction.")
        
    # 2. Convert raw data to DataFrame for processing
    input_df = pd.DataFrame([raw_data])
    
    # 3. Feature Engineering and Time-Based Processing
    input_df['created_at'] = pd.to_datetime(input_df['created_at'])
    input_df['hours'] = input_df['created_at'].dt.hour
    input_df['day'] = input_df['created_at'].dt.dayofweek
    
    # Rush Hour and Weekend Flags
    input_df['is_lunch_rush'] = input_df['hours'].apply(lambda x: 1 if 11 <= x <= 14 else 0)
    input_df['is_dinner_rush'] = input_df['hours'].apply(lambda x: 1 if 17 <= x <= 21 else 0)
    input_df['is_weekend'] = input_df['day'].apply(lambda x: 1 if x >= 5 else 0)

    # Handle Historical Average Feature (Fallback)
    if 'store_avg_delivery_time' not in input_df.columns:
         input_df['store_avg_delivery_time'] = GLOBAL_MEAN_TIME
    
    # 4. One-Hot Encoding (OHE)
    input_df = pd.get_dummies(input_df, columns=['store_primary_category'])
    
    # 5. Final Feature Alignment (CRITICAL FOR DEPLOYMENT)
    # Reindex to ensure the input features are in the exact order and format (columns)
    # expected by the trained model. Fill missing OHE columns with 0.
    final_features = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # 6. Scaling
    x_scaled = scaler.transform(final_features)
    
    # 7. Prediction (Model must predict on a numpy array)
    prediction = model.predict(x_scaled)[0][0]
    
    return float(prediction)


# --- 3. Initialize FastAPI and Define Schema ---

app = FastAPI(
    title="Delivery Time Prediction API",
    description="NN model to estimate order delivery time."
)

class OrderInput(BaseModel):
    # Base features must match the expected API request structure
    market_id: float
    created_at: str 
    subtotal: int
    total_items: int
    num_distinct_items: int
    min_item_price: int
    max_item_price: int
    total_onshift_partners: float
    total_busy_partners: float
    total_outstanding_orders: float
    store_primary_category: str
    store_avg_delivery_time: float # Assumed to be retrieved by caller


# --- 4. Define API Endpoint ---

@app.post("/predict_time")
def predict(order_data: OrderInput):
    # Prepare the dictionary for the pipeline
    raw_data = order_data.dict()
    
    try:
        prediction = predict_delivery_time(
            raw_data=raw_data,
            model=final_model,
            scaler=fitted_scaler,
            feature_columns=FINAL_COLUMNS
        )
        
        return {
            "status": "success",
            "estimated_delivery_minutes": round(prediction, 2)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Prediction pipeline failed: {str(e)}"}

# --- END OF SCRIPT ---