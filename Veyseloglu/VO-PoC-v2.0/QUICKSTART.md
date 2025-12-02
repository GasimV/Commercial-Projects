# Quick Setup Guide

## Installation (5 minutes)

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python -c "import tensorflow; import lightgbm; import fastapi; print('✅ All packages installed')"
```

## Running the System (2 options)

### Option A: Automated Startup (Recommended)
```bash
python start.py
```
This will:
- Check dependencies
- Start API server on port 8000
- Start frontend server on port 8080
- Open browser automatically

### Option B: Manual Startup
```bash
# Terminal 1: Start API
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Frontend
cd app/static
python -m http.server 8080

# Open browser
# Navigate to http://localhost:8080
```

## Using the System

### Step 1: Upload Data
- Drag & drop your CSV file or click to browse
- Required columns: DATE, Partner Customer Code, Product Code, NetSalesQty
- Wait for data summary to appear

### Step 2: Train Models
- Set prediction horizon (default: 14 days)
- Click "Train All Models"
- Wait 5-15 minutes for training to complete
- Review performance metrics

### Step 3: Get Predictions
Choose one of three tabs:

**By Customer:**
- Enter customer ID
- Select model (Ensemble recommended)
- View top products likely to be reordered

**By Product:**
- Enter product ID
- View top customers likely to reorder

**Compare Models:**
- Enter customer ID
- See predictions from all models side-by-side

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Upload Data
```bash
curl -X POST -F "file=@yourdata.csv" http://localhost:8000/upload_data
```

### Train Models
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"prediction_horizon": 14}'
```

### Get Predictions
```bash
# For a customer
curl "http://localhost:8000/predict/customer/1000000001?model=ensemble&top_k=20"

# For a product
curl "http://localhost:8000/predict/product/000000001000005732?model=ensemble&top_k=20"
```

## Troubleshooting

### Port Already in Use
If ports 8000 or 8080 are already in use:
```bash
# API server on different port
python -m uvicorn app.main:app --port 8001

# Frontend on different port
cd app/static && python -m http.server 8081
```

### Module Not Found
Ensure you're in the project root directory:
```bash
cd advanced_reorder_poc
python start.py
```

### Out of Memory
Reduce batch size in training:
- Edit `app/models/training_pipeline.py`
- Change `batch_size=256` to `batch_size=128` or lower

### Training Takes Too Long
- Use GPU if available (TensorFlow will auto-detect)
- Reduce data size for testing
- Use LightGBM only (fastest model)

## Next Steps

1. **Explore the Analysis Notebook:**
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

2. **Check API Documentation:**
   - Visit http://localhost:8000/docs
   - Interactive API testing interface

3. **Monitor Performance:**
   - Check `models_store/` for saved models
   - Review metrics in `*_metrics.json` files

4. **Production Deployment:**
   - Add authentication
   - Set up model versioning
   - Implement monitoring
   - Schedule automated retraining

## Support

- Check logs in the terminal for detailed errors
- Review README.md for comprehensive documentation
- API docs: http://localhost:8000/docs
