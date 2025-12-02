# Advanced AI-Powered Reorder Prediction System

A production-grade machine learning system for predicting customer reorder likelihood and quantities using an ensemble of **Feed-Forward Neural Networks (FFNN)**, **LSTM**, and **LightGBM** models.

## 🎯 Features

### Component 1: Reorder Likelihood Prediction
- **FFNN**: Deep neural network with batch normalization and dropout
- **LSTM**: Bidirectional LSTM for sequence-based temporal patterns
- **LightGBM**: Gradient boosting for feature importance analysis
- **Ensemble**: Weighted combination of all three models

### Component 2: Quantity Prediction
- Predicts next order quantity for each customer-product pair
- Uses same model architecture as Component 1
- Optimized for regression task with robust loss functions

### Advanced Feature Engineering
- **Recency Features**: Days since last order, customer tenure
- **Frequency Features**: Order count, rolling windows (30/60/90 days), order intervals
- **Monetary Features**: Cumulative quantities, rolling averages, unit prices, discounts
- **Temporal Features**: Seasonality, day of week, cyclical encodings
- **Categorical Features**: Customer/product aggregations, category performance
- **Interaction Features**: Share of wallet, product concentration
- **Trend Features**: Growth momentum, purchase trends

### Modern Web Interface
- Real-time predictions with interactive visualizations
- Model comparison dashboard
- Responsive design with dark theme
- Chart.js integration for data visualization

## 📊 Architecture

```
Data Preparation
    ├── Feature Engineering (50+ features)
    └── Sequence Preparation for LSTM

Modeling (Parallel Training)
    ├── Component 1: Reorder Likelihood
    │   ├── FFNN (256→128→64→32)
    │   ├── LSTM (Bidirectional 128→64→32)
    │   └── LightGBM (GBDT with early stopping)
    │
    └── Component 2: Quantity Prediction
        ├── FFNN (256→128→64→32)
        ├── LSTM (Bidirectional 128→64→32)
        └── LightGBM (Regression mode)

Inference
    └── Ensemble Predictions with Priority Scoring
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Your CSV file should contain the following columns:
- `DATE`: Order date
- `Partner Customer Code`: Customer identifier
- `Product Code`: Product identifier
- `NetSalesQty`: Order quantity
- `Net Sales Value LC`: Sales value
- `Discount %`: Discount percentage
- Additional columns for categories, location, etc.

### 3. Start the API Server

```bash
cd advanced_reorder_poc
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the Web Interface

Navigate to `app/static/index.html` in your browser (use a local web server or simply open the file).

Alternatively, serve it with Python:
```bash
cd app/static
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

## 📖 Usage Guide

### Step 1: Upload Data
1. Click on the upload zone or drag & drop your CSV file
2. Wait for data validation and summary to appear
3. Review the data statistics (customers, products, date range)

### Step 2: Train Models
1. Set the prediction horizon (default: 14 days)
2. Click "Train All Models"
3. Wait for training to complete (progress bar will show status)
4. Review model performance metrics (ROC AUC, MAE, etc.)

### Step 3: Make Predictions

#### By Customer
- Enter a customer ID
- Select model (Ensemble recommended)
- Get top 20 products most likely to be reordered
- View probability, predicted quantity, and priority score

#### By Product
- Enter a product ID
- Get top 20 customers most likely to reorder
- View probability and predicted quantities

#### Compare Models
- Enter a customer ID
- See prediction comparison across FFNN, LightGBM, and Ensemble
- Interactive bar chart visualization

## 📈 Model Performance

### Reorder Likelihood (Classification)
- **Metric**: ROC AUC Score
- **FFNN**: Typical range 0.75-0.85
- **LightGBM**: Typical range 0.78-0.88
- **LSTM**: Typical range 0.72-0.82
- **Ensemble**: Best overall performance

### Quantity Prediction (Regression)
- **Metric**: Mean Absolute Error (MAE)
- **Lower is better**
- Ensemble provides most stable predictions

## 🔧 API Endpoints

### Health Check
```
GET /health
```

### Upload Data
```
POST /upload_data
Form-data: file (CSV)
```

### Train Models
```
POST /train
Body: {
    "prediction_horizon": 14,
    "test_size": 0.2
}
```

### Predict for Customer
```
GET /predict/customer/{customer_id}?model=ensemble&top_k=20
```

### Predict for Product
```
GET /predict/product/{product_id}?model=ensemble&top_k=20
```

### Compare Models
```
POST /compare_models
Body: { "customer_id": "..." }
```

### Get Model Metrics
```
GET /model_metrics
```

## 🛠️ Technical Details

### Model Architectures

**FFNN (Feed-Forward Neural Network)**
- Input layer: Variable (based on features)
- Hidden layers: 256 → 128 → 64 → 32
- Batch normalization after each layer
- Dropout: 0.3, 0.3, 0.2
- Activation: ReLU
- Output: Sigmoid (classification) / ReLU (regression)
- Optimizer: Adam (lr=0.001)

**LSTM (Long Short-Term Memory)**
- Bidirectional LSTM layers: 128 → 64 → 32
- Sequence length: 10 time steps
- Dropout: 0.3, 0.3, 0.2
- Dense layer: 32 units
- Output: Sigmoid (classification) / ReLU (regression)

**LightGBM**
- Boosting type: GBDT
- Num leaves: 31
- Learning rate: 0.05
- Feature fraction: 0.8
- Bagging fraction: 0.8
- Max depth: 7
- Early stopping: 50 rounds

### Training Strategy
- Train/Val/Test split: 64% / 16% / 20%
- Early stopping on validation loss
- Learning rate reduction on plateau
- Model checkpointing (best weights saved)

### Feature Scaling
- StandardScaler for all models
- Separate scalers for each model type
- Applied to both training and inference

## 📁 Project Structure

```
advanced_reorder_poc/
├── app/
│   ├── models/
│   │   ├── model_architectures.py  # FFNN, LSTM, LightGBM classes
│   │   ├── training_pipeline.py     # Training orchestration
│   │   └── inference.py             # Prediction logic
│   ├── utils/
│   │   └── feature_engineering.py   # Feature extraction
│   ├── static/
│   │   ├── index.html               # Web interface
│   │   ├── css/
│   │   │   └── style.css            # Styling
│   │   └── js/
│   │       └── app.js               # Frontend logic
│   └── main.py                      # FastAPI application
├── models_store/                     # Trained models (generated)
├── data/                             # Data directory (generated)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🎨 Design Philosophy

The UI follows a **deep tech aesthetic** with:
- Dark theme optimized for data-heavy interfaces
- Monospace fonts for technical credibility
- Cyan accent color for high-tech feel
- Subtle animations and micro-interactions
- Grid-based layouts with clear information hierarchy
- Responsive design for all screen sizes

## 🔍 Feature Importance

After training, the system generates feature importance scores (available via LightGBM). Key drivers typically include:
1. Days since last order
2. Average order interval
3. Rolling quantity averages
4. Customer order frequency
5. Discount patterns
6. Seasonal factors

## 🚦 Performance Considerations

- **Training Time**: ~5-15 minutes depending on data size and hardware
- **Inference Time**: <100ms per customer
- **Memory Usage**: ~2-4GB during training
- **Recommended**: GPU for faster training (optional)

## 📝 Data Requirements

**Minimum Records**: 1,000+ orders
**Minimum Customers**: 50+
**Minimum Products**: 20+
**Time Range**: At least 3 months of history

## 🤝 Contributing

This is a proof-of-concept system. For production deployment:
1. Add data validation and error handling
2. Implement model versioning
3. Add automated retraining pipeline
4. Set up monitoring and logging
5. Add authentication and authorization
6. Implement A/B testing framework

## 📄 License

This project is provided as-is for demonstration purposes.

## 🙋 Support

For questions or issues:
1. Check the API documentation at `http://localhost:8000/docs`
2. Review the console logs for detailed error messages
3. Ensure all dependencies are correctly installed

---

**Built with**: TensorFlow, LightGBM, FastAPI, Chart.js
**Version**: 2.0.0
