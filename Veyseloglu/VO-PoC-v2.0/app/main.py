"""
FastAPI Backend for Advanced Reorder Prediction System
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import os
import io
import json
import traceback

from app.models.training_pipeline import ReorderTrainingPipeline, QuantityTrainingPipeline
from app.models.inference import ReorderPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Reorder & Quantity Prediction API",
    description="AI-powered system for predicting customer reorders and quantities",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
DATA_DIR = "data"
MODEL_DIR = "models_store"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Global data holder
current_data = None
predictor = None


class TrainRequest(BaseModel):
    prediction_horizon: int = 30
    test_size: float = 0.2
    resume_training: bool = False

class CompareRequest(BaseModel):
    customer_id: str

class PredictionResponse(BaseModel):
    customer_id: str
    product_id: str
    reorder_probability: float
    predicted_quantity: int
    priority_score: float


@app.on_event("startup")
async def startup_event():
    """Load models on startup if they exist"""
    global predictor

    if os.path.exists(os.path.join(MODEL_DIR, 'reorder_likelihood_lgbm_model.txt')):
        predictor = ReorderPredictor(MODEL_DIR)
        try:
            predictor.load_reorder_models()
            predictor.load_quantity_models()
            print("✓ Models loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load models: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced Reorder Prediction API",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "data_loaded": current_data is not None,
        "models_loaded": predictor is not None
    }


@app.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload CSV or Parquet data file
    """
    global current_data

    try:
        contents = await file.read()
        filename = (file.filename or "").lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".parquet", ".pq")):
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(
                400,
                "Unsupported file type. Please upload a CSV or Parquet file."
            )

        # Validate required columns
        required_cols = ['DATE', 'Partner Customer Code', 'Product Code', 'NetSalesQty']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise HTTPException(400, f"Missing required columns: {missing}")

        current_data = df

        # Save a canonical version to disk (still as CSV, or as Parquet if you prefer)
        df.to_csv(os.path.join(DATA_DIR, "sales_data.csv"), index=False)

        return {
            "status": "success",
            "message": "Data uploaded successfully",
            "rows": len(df),
            "columns": len(df.columns),
            "date_range": {
                "start": str(pd.to_datetime(df['DATE']).min()),
                "end": str(pd.to_datetime(df['DATE']).max())
            },
            "unique_customers": int(df['Partner Customer Code'].nunique()),
            "unique_products": int(df['Product Code'].nunique())
        }

    except Exception as e:
        raise HTTPException(500, f"Error processing file: {str(e)}")


@app.get("/data_summary")
async def data_summary():
    """
    Get summary statistics of loaded data
    """
    global current_data

    if current_data is None:
        raise HTTPException(400, "No data loaded. Please upload data first.")

    df = current_data.copy()
    df['DATE'] = pd.to_datetime(df['DATE'])

    return {
        "total_records": len(df),
        "date_range": {
            "start": str(df['DATE'].min()),
            "end": str(df['DATE'].max()),
            "days": int((df['DATE'].max() - df['DATE'].min()).days)
        },
        "customers": {
            "total": int(df['Partner Customer Code'].nunique()),
            "top_10": df['Partner Customer Code'].value_counts().head(10).to_dict()
        },
        "products": {
            "total": int(df['Product Code'].nunique()),
            "categories": df['Product Level H1'].value_counts().to_dict() if 'Product Level H1' in df.columns else {}
        },
        "sales": {
            "total_quantity": float(df['NetSalesQty'].sum()),
            "total_value": float(df['Net Sales Value LC'].sum()) if 'Net Sales Value LC' in df.columns else 0,
            "avg_order_qty": float(df['NetSalesQty'].mean())
        }
    }


@app.post("/train")
async def train_models(request: TrainRequest):
    """
    Train all models (FFNN, LSTM, LightGBM)
    """
    global current_data, predictor

    if current_data is None:
        raise HTTPException(400, "No data loaded. Please upload data first.")

    try:
        print("\n" + "=" * 80)
        print(f"STARTING MODEL TRAINING (Horizon: {request.prediction_horizon} days, Resume: {request.resume_training})")
        print("=" * 80)

        # Train reorder likelihood models
        print("\n### COMPONENT 1: REORDER LIKELIHOOD ###")
        reorder_pipeline = ReorderTrainingPipeline(MODEL_DIR, DATA_DIR, prediction_horizon=request.prediction_horizon)
        reorder_metrics = reorder_pipeline.train_all(current_data, resume_training=request.resume_training)

        # Train quantity prediction models
        print("\n### COMPONENT 2: QUANTITY PREDICTION ###")
        quantity_pipeline = QuantityTrainingPipeline(MODEL_DIR, DATA_DIR, prediction_horizon=request.prediction_horizon)
        quantity_metrics = quantity_pipeline.train_all(current_data, resume_training=request.resume_training)

        # Save training configuration metadata
        training_config = {
            "prediction_horizon": request.prediction_horizon,
            "test_size": request.test_size,
            "resume_training": request.resume_training
        }
        with open(os.path.join(MODEL_DIR, 'training_config.json'), 'w') as f:
            json.dump(training_config, f, indent=2)

        # Load trained models into predictor
        predictor = ReorderPredictor(MODEL_DIR)
        predictor.load_reorder_models()
        predictor.load_quantity_models()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)

        return {
            "status": "success",
            "message": f"All models trained successfully with {request.prediction_horizon}-day horizon",
            "prediction_horizon": request.prediction_horizon,
            "reorder_likelihood_metrics": reorder_metrics,
            "quantity_prediction_metrics": quantity_metrics
        }

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR DURING TRAINING")
        print("=" * 80)
        print(repr(e))
        traceback.print_exc()
        print("=" * 80 + "\n")
        # Still return 500 to the frontend
        raise HTTPException(500, f"Training failed: {str(e)}")


@app.get("/predict/customer/{customer_id}")
async def predict_for_customer(
        customer_id: str,
        model: str = Query("ensemble", description="Model to use: ffnn, lgbm, or ensemble"),
        top_k: int = Query(20, description="Number of products to return"),
        min_probability: float = Query(0.5, ge=0.0, le=1.0, description="Minimum reorder probability threshold (0-1)")
):
    """
    Get predictions for a specific customer

    Args:
        customer_id: Customer ID to predict for
        model: Model to use for predictions
        top_k: Maximum number of results to return
        min_probability: Only return predictions with reorder_probability >= this threshold
                        (Recommended: 0.7-0.85 for conditional quantity predictions)
    """
    global current_data, predictor

    if current_data is None:
        raise HTTPException(400, "No data loaded")

    if predictor is None:
        raise HTTPException(400, "Models not trained. Please train models first.")

    try:
        predictions = predictor.predict_for_customer(
            current_data, customer_id, model, top_k, min_probability
        )

        if len(predictions) == 0:
            return {
                "message": f"No predictions available for customer {customer_id}",
                "predictions": []
            }

        return {
            "customer_id": customer_id,
            "model_used": model,
            "total_products": len(predictions),
            "predictions": predictions.to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.get("/predict/product/{product_id}")
async def predict_for_product(
        product_id: str,
        model: str = Query("ensemble", description="Model to use: ffnn, lgbm, or ensemble"),
        top_k: int = Query(20, description="Number of customers to return"),
        min_probability: float = Query(0.5, ge=0.0, le=1.0, description="Minimum reorder probability threshold (0-1)")
):
    """
    Get predictions for a specific product (which customers will reorder)

    Args:
        product_id: Product ID to predict for
        model: Model to use for predictions
        top_k: Maximum number of results to return
        min_probability: Only return predictions with reorder_probability >= this threshold
                        (Recommended: 0.7-0.85 for conditional quantity predictions)
    """
    global current_data, predictor

    if current_data is None:
        raise HTTPException(400, "No data loaded")

    if predictor is None:
        raise HTTPException(400, "Models not trained")

    try:
        predictions = predictor.predict_for_product(
            current_data, product_id, model, top_k, min_probability
        )

        if len(predictions) == 0:
            return {
                "message": f"No predictions available for product {product_id}",
                "predictions": []
            }

        return {
            "product_id": product_id,
            "model_used": model,
            "total_customers": len(predictions),
            "predictions": predictions.to_dict(orient='records')
        }

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.get("/customers")
async def get_customers(limit: int = Query(100, description="Number of customers to return")):
    """
    Get list of customers
    """
    global current_data

    if current_data is None:
        raise HTTPException(400, "No data loaded")

    customers = current_data['Partner Customer Code'].value_counts().head(limit)

    return {
        "total_customers": int(current_data['Partner Customer Code'].nunique()),
        "customers": [
            {
                "customer_id": str(cust_id),
                "order_count": int(count)
            }
            for cust_id, count in customers.items()
        ]
    }


@app.get("/products")
async def get_products(
        limit: int = Query(100, description="Number of products to return"),
        category: Optional[str] = Query(None, description="Filter by category")
):
    """
    Get list of products
    """
    global current_data

    if current_data is None:
        raise HTTPException(400, "No data loaded")

    df = current_data.copy()

    if category and 'Product Level H1' in df.columns:
        df = df[df['Product Level H1'] == category]

    products = df['Product Code'].value_counts().head(limit)

    # Get product names if available
    product_info = []
    for prod_id, count in products.items():
        prod_data = df[df['Product Code'] == prod_id].iloc[0]
        product_info.append({
            "product_id": str(prod_id),
            "product_name": str(prod_data.get('Product Name', 'N/A')),
            "category": str(prod_data.get('Product Level H1', 'N/A')),
            "order_count": int(count)
        })

    return {
        "total_products": int(df['Product Code'].nunique()),
        "products": product_info
    }


@app.get("/categories")
async def get_categories():
    """
    Get list of product categories
    """
    global current_data

    if current_data is None:
        raise HTTPException(400, "No data loaded")

    if 'Product Level H1' not in current_data.columns:
        return {"categories": []}

    categories = current_data['Product Level H1'].value_counts()

    return {
        "categories": [
            {
                "name": str(cat),
                "product_count": int(count)
            }
            for cat, count in categories.items()
        ]
    }


@app.get("/model_metrics")
async def get_model_metrics():
    """
    Get performance metrics of trained models
    """
    reorder_metrics_path = os.path.join(MODEL_DIR, 'reorder_likelihood_metrics.json')
    quantity_metrics_path = os.path.join(MODEL_DIR, 'quantity_prediction_metrics.json')

    if not os.path.exists(reorder_metrics_path):
        raise HTTPException(400, "Models not trained yet")

    with open(reorder_metrics_path, 'r') as f:
        reorder_metrics = json.load(f)

    with open(quantity_metrics_path, 'r') as f:
        quantity_metrics = json.load(f)

    return {
        "reorder_likelihood": reorder_metrics,
        "quantity_prediction": quantity_metrics
    }


@app.post("/compare_models")
async def compare_models(request: CompareRequest):
    """
    Compare predictions from different models for a customer
    """
    customer_id = request.customer_id
    global current_data, predictor

    if current_data is None or predictor is None:
        raise HTTPException(400, "Data not loaded or models not trained")

    try:
        from app.utils.feature_engineering import get_latest_features_for_inference
        import math # Import math to check for NaN

        latest = get_latest_features_for_inference(current_data, customer_id=customer_id)

        if len(latest) == 0:
            raise HTTPException(404, f"No data found for customer {customer_id}")

        # 1. FIX INDEX: Reset index so row iteration matches array indices [0, 1, 2...]
        latest = latest.reset_index(drop=True)

        feature_cols = predictor.engineer.get_feature_columns()['all']
        X = latest[feature_cols].fillna(0).values

        # Get predictions from all models
        comparisons = predictor.get_model_comparison(X)

        # Format results
        results = []
        for i, row in latest.iterrows():
            product_comparison = {
                "product_id": str(row['product_id']),
                "reorder_likelihood": {},
                "quantity_prediction": {}
            }

            # Helper function to safely convert numpy/NaN to valid float
            def safe_float(val):
                try:
                    f_val = float(val)
                    if math.isnan(f_val) or math.isinf(f_val):
                        return 0.0
                    return f_val
                except:
                    return 0.0

            # 2. FILL REORDER PREDICTIONS
            for model_name in comparisons['reorder']:
                # Safe access with array bounds check
                arr = comparisons['reorder'][model_name]
                val = arr[i] if i < len(arr) else 0.0
                product_comparison['reorder_likelihood'][model_name] = safe_float(val)

            # 3. FILL QUANTITY PREDICTIONS
            for model_name in comparisons['quantity']:
                arr = comparisons['quantity'][model_name]
                val = arr[i] if i < len(arr) else 0.0
                product_comparison['quantity_prediction'][model_name] = safe_float(val)

            results.append(product_comparison)

        # FIX Q4: Sort by Ensemble Reorder Probability Descending
        def get_sort_key(item):
            # Try to get ensemble score, default to 0
            return item['reorder_likelihood'].get('ensemble', 0)

        results.sort(key=get_sort_key, reverse=True)

        return {
            "customer_id": customer_id,
            "comparisons": results[:10]  # Now this returns the TOP 10 highest probability items
        }


    except Exception as e:
        # 4. DEBUG LOGGING: This will print the EXACT error to your VS Code terminal
        print("\n" + "!"*30)
        print(f"COMPARE MODELS ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("!"*30 + "\n")

        raise HTTPException(500, f"Comparison failed: {str(e)}")


@app.get("/predict/stock_forecast")
async def get_stock_forecast(
    min_probability: float = Query(0.5, description="Only count orders with prob > X"),
    fabric_filter: Optional[str] = Query(None, description="Filter by Manufacturer/Fabric")
):
    global current_data, predictor
    if current_data is None or predictor is None:
        raise HTTPException(400, "System not ready. Please upload data and train models.")

    try:
        from app.utils.feature_engineering import get_latest_features_for_inference

        # 1. Get snapshot of ALL customers
        latest = get_latest_features_for_inference(current_data)

        if len(latest) == 0:
            return {"count": 0, "forecast": []}

        # --- FIX: ROBUST MERGE START ---
        # 1. Strip whitespace from column names to prevent KeyErrors
        # e.g., "Product Name " -> "Product Name"
        current_data.columns = current_data.columns.str.strip()

        # 2. Define required columns
        meta_cols = ['Product Code', 'Product Name', 'Product Manufacturer']

        # 3. Create metadata dataframe safely
        # Check which columns actually exist in raw data
        available_cols = [c for c in meta_cols if c in current_data.columns]

        product_meta = current_data[available_cols].drop_duplicates('Product Code')

        # 4. Convert IDs to string for reliable merging
        if 'Product Code' in product_meta.columns:
            product_meta['Product Code'] = product_meta['Product Code'].astype(str)
        latest['product_id'] = latest['product_id'].astype(str)

        # 5. Clean 'latest' before merge
        # Remove any metadata columns if they somehow exist to avoid conflicts
        latest = latest.drop(columns=[c for c in available_cols if c in latest.columns], errors='ignore')

        # 6. Merge
        if 'Product Code' in product_meta.columns:
            latest = latest.merge(
                product_meta,
                left_on='product_id',
                right_on='Product Code',
                how='left'
            )

        # 7. Fill Missing/Non-existent columns with Defaults
        if 'Product Manufacturer' not in latest.columns:
            latest['Product Manufacturer'] = 'Unknown'
        else:
            latest['Product Manufacturer'] = latest['Product Manufacturer'].fillna('Unknown')

        if 'Product Name' not in latest.columns:
            latest['Product Name'] = 'Unknown Product'
        else:
            latest['Product Name'] = latest['Product Name'].fillna('Unknown Product')
        # --- FIX END ---

        # 2. Filter by Fabric/Manufacturer if requested
        if fabric_filter:
            mask = latest['Product Manufacturer'].astype(str).str.lower().str.contains(fabric_filter.lower())
            latest = latest[mask]

        if len(latest) == 0:
            return {"count": 0, "forecast": []}

        # 3. Predict
        feature_cols = predictor.engineer.get_feature_columns()['all']
        X = latest[feature_cols].fillna(0).values

        probs = predictor.predict_reorder_likelihood(X, 'lgbm')
        qtys = predictor.predict_quantity(X, 'lgbm')

        # 4. Create DataFrame
        forecast_df = pd.DataFrame({
            'Product Code': latest['product_id'],
            'Product Name': latest['Product Name'],
            'Manufacturer': latest['Product Manufacturer'],
            'Probability': probs,
            'Expected_Qty': qtys
        })

        # 5. Filter & Aggregate
        likely_orders = forecast_df[forecast_df['Probability'] >= min_probability]

        stock_plan = likely_orders.groupby(['Product Code', 'Product Name', 'Manufacturer']).agg(
            Total_Qty=('Expected_Qty', 'sum'),
            Customer_Count=('Product Code', 'count'),
            Avg_Confidence=('Probability', 'mean')
        ).reset_index()

        # 6. Sort
        stock_plan = stock_plan.sort_values('Total_Qty', ascending=False).head(50)

        return {
            "count": len(stock_plan),
            "forecast": stock_plan.to_dict(orient='records')
        }

    except Exception as e:
        print(f"Stock Forecast Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Forecast failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)