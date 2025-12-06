"""
Inference Module for Reorder Predictions
"""

import numpy as np
import pandas as pd
import os
import joblib
from typing import Dict, List, Tuple
from tensorflow import keras

from app.utils.feature_engineering import FeatureEngineer, get_latest_features_for_inference
from app.models.model_architectures import LightGBMModel


class ReorderPredictor:
    """
    Unified predictor for reorder likelihood and quantity
    """

    def __init__(self, model_dir: str = 'models_store'):
        self.model_dir = model_dir
        self.engineer = FeatureEngineer(prediction_horizon=14)

        # Reorder likelihood models
        self.reorder_models = {
            'ffnn': None,
            'lgbm': None,
            'lstm': None
        }
        self.reorder_scalers = {}

        # Quantity models
        self.quantity_models = {
            'ffnn': None,
            'lgbm': None,
            'lstm': None
        }
        self.quantity_scalers = {}

    def load_reorder_models(self):
        """Load reorder likelihood models"""
        print("Loading reorder likelihood models...")

        # Load FFNN
        try:
            ffnn_path = os.path.join(self.model_dir, 'reorder_likelihood_ffnn')
            self.reorder_models['ffnn'] = keras.models.load_model(f"{ffnn_path}_model.h5")
            self.reorder_scalers['ffnn'] = joblib.load(f"{ffnn_path}_scaler.pkl")
            print("✓ FFNN loaded")
        except Exception as e:
            print(f"✗ FFNN failed: {e}")

        # Load LightGBM
        try:
            lgbm_path = os.path.join(self.model_dir, 'reorder_likelihood_lgbm')
            import lightgbm as lgb
            self.reorder_models['lgbm'] = lgb.Booster(model_file=f"{lgbm_path}_model.txt")
            self.reorder_scalers['lgbm'] = joblib.load(f"{lgbm_path}_scaler.pkl")
            print("✓ LightGBM loaded")
        except Exception as e:
            print(f"✗ LightGBM failed: {e}")

        # Load LSTM
        try:
            lstm_path = os.path.join(self.model_dir, 'reorder_likelihood_lstm')
            self.reorder_models['lstm'] = keras.models.load_model(f"{lstm_path}_model.h5")
            self.reorder_scalers['lstm'] = joblib.load(f"{lstm_path}_scaler.pkl")
            print("✓ LSTM loaded")
        except Exception as e:
            print(f"✗ LSTM failed: {e}")

    def load_quantity_models(self):
        """Load quantity prediction models"""
        print("Loading quantity prediction models...")

        # Load FFNN
        try:
            ffnn_path = os.path.join(self.model_dir, 'quantity_prediction_ffnn')
            self.quantity_models['ffnn'] = keras.models.load_model(f"{ffnn_path}_model.h5")
            self.quantity_scalers['ffnn'] = joblib.load(f"{ffnn_path}_scaler.pkl")
            print("✓ FFNN loaded")
        except Exception as e:
            print(f"✗ FFNN failed: {e}")

        # Load LightGBM
        try:
            lgbm_path = os.path.join(self.model_dir, 'quantity_prediction_lgbm')
            import lightgbm as lgb
            self.quantity_models['lgbm'] = lgb.Booster(model_file=f"{lgbm_path}_model.txt")
            self.quantity_scalers['lgbm'] = joblib.load(f"{lgbm_path}_scaler.pkl")
            print("✓ LightGBM loaded")
        except Exception as e:
            print(f"✗ LightGBM failed: {e}")

        # Load LSTM
        try:
            lstm_path = os.path.join(self.model_dir, 'quantity_prediction_lstm')
            self.quantity_models['lstm'] = keras.models.load_model(f"{lstm_path}_model.h5")
            self.quantity_scalers['lstm'] = joblib.load(f"{lstm_path}_scaler.pkl")
            print("✓ LSTM loaded")
        except Exception as e:
            print(f"✗ LSTM failed: {e}")

    def predict_reorder_likelihood(self, X: np.ndarray,
                                   model_name: str = 'ensemble') -> np.ndarray:
        """
        Predict reorder likelihood

        Args:
            X: Feature matrix
            model_name: 'ffnn', 'lgbm', 'lstm', or 'ensemble'

        Returns:
            Probability array
        """
        if model_name == 'ensemble':
            predictions = []
            weights = []

            # FFNN
            if self.reorder_models['ffnn'] is not None:
                X_scaled = self.reorder_scalers['ffnn'].transform(X)
                pred = self.reorder_models['ffnn'].predict(X_scaled, verbose=0).flatten()
                predictions.append(pred)
                weights.append(0.33)

            # LightGBM
            if self.reorder_models['lgbm'] is not None:
                X_scaled = self.reorder_scalers['lgbm'].transform(X)
                pred = self.reorder_models['lgbm'].predict(X_scaled)
                predictions.append(pred)
                weights.append(0.33)

            # LSTM (skip for now in tabular inference)
            # Would need sequence data

            # Weighted average
            weights = np.array(weights) / sum(weights)
            ensemble_pred = sum(p * w for p, w in zip(predictions, weights))
            return ensemble_pred

        else:
            # Single model prediction
            if self.reorder_models[model_name] is None:
                raise ValueError(f"Model {model_name} not loaded")

            X_scaled = self.reorder_scalers[model_name].transform(X)

            if model_name == 'lgbm':
                return self.reorder_models[model_name].predict(X_scaled)
            else:
                return self.reorder_models[model_name].predict(X_scaled, verbose=0).flatten()

    def predict_quantity(self, X: np.ndarray,
                        model_name: str = 'ensemble') -> np.ndarray:
        """
        Predict next order quantity

        Args:
            X: Feature matrix
            model_name: 'ffnn', 'lgbm', 'lstm', or 'ensemble'

        Returns:
            Quantity predictions
        """
        if model_name == 'ensemble':
            predictions = []
            weights = []

            # FFNN
            if self.quantity_models['ffnn'] is not None:
                X_scaled = self.quantity_scalers['ffnn'].transform(X)
                pred = self.quantity_models['ffnn'].predict(X_scaled, verbose=0).flatten()
                predictions.append(pred)
                weights.append(0.5)

            # LightGBM
            if self.quantity_models['lgbm'] is not None:
                X_scaled = self.quantity_scalers['lgbm'].transform(X)
                pred = self.quantity_models['lgbm'].predict(X_scaled)
                predictions.append(pred)
                weights.append(0.5)

            # Weighted average
            weights = np.array(weights) / sum(weights)
            ensemble_pred = sum(p * w for p, w in zip(predictions, weights))
            return ensemble_pred

        else:
            # Single model prediction
            if self.quantity_models[model_name] is None:
                raise ValueError(f"Model {model_name} not loaded")

            X_scaled = self.quantity_scalers[model_name].transform(X)

            if model_name == 'lgbm':
                return self.quantity_models[model_name].predict(X_scaled)
            else:
                return self.quantity_models[model_name].predict(X_scaled, verbose=0).flatten()

    def predict_for_customer(self, df: pd.DataFrame,
                             customer_id: str,
                             model_name: str = 'ensemble',
                             top_k: int = 20) -> pd.DataFrame:

        # 1. Get predictions (existing logic)
        latest = get_latest_features_for_inference(df, customer_id=customer_id)
        if len(latest) == 0: return pd.DataFrame()
        latest = latest.reset_index(drop=True)

        feature_cols = self.engineer.get_feature_columns()['all']
        X = latest[feature_cols].fillna(0).values

        reorder_probs = self.predict_reorder_likelihood(X, model_name)
        quantities = self.predict_quantity(X, model_name)

        # 2. CREATE LOOKUP FOR NAMES (New Logic)
        # We create a small lookup df from the original raw data to get names
        # We drop duplicates to get unique Product Code -> Name mappings
        product_lookup = df[['Product Code', 'Product Name', 'Product Manufacturer']].drop_duplicates('Product Code')
        # Ensure types match for merging
        product_lookup['Product Code'] = product_lookup['Product Code'].astype(str)
        latest['product_id'] = latest['product_id'].astype(str)

        # 3. BUILD RESULTS
        results = pd.DataFrame({
            'customer_id': latest['customer_id'].values,
            'product_id': latest['product_id'].values,
            'category': latest.get('category_h1', ['N/A'] * len(latest)).values,
            'reorder_probability': reorder_probs,
            'predicted_quantity': np.maximum(0, np.round(quantities)).astype(int),
            'days_since_last_order': latest['days_since_last_order'].values,
            'avg_discount': latest.get('avg_discount', [0]*len(latest)).values
        })

        # 4. MERGE NAMES
        results = results.merge(
            product_lookup,
            left_on='product_id',
            right_on='Product Code',
            how='left'
        )

        # Fill missing names
        results['Product Name'] = results['Product Name'].fillna('Unknown Product')
        results['Product Manufacturer'] = results['Product Manufacturer'].fillna('')

        # 5. SCORE & SORT
        results['priority_score'] = (
            results['reorder_probability'] * 0.6 +
            (results['predicted_quantity'] / (results['predicted_quantity'].max() + 1)) * 0.4
        )
        results = results.sort_values('priority_score', ascending=False)

        return results.head(top_k)

    def predict_for_product(self, df: pd.DataFrame,
                            product_id: str,
                            model_name: str = 'ensemble',
                            top_k: int = 20) -> pd.DataFrame:

        latest = get_latest_features_for_inference(df, product_id=product_id)
        if len(latest) == 0: return pd.DataFrame()
        latest = latest.reset_index(drop=True)

        feature_cols = self.engineer.get_feature_columns()['all']
        X = latest[feature_cols].fillna(0).values

        reorder_probs = self.predict_reorder_likelihood(X, model_name)
        quantities = self.predict_quantity(X, model_name)

        # 1. CREATE LOOKUP FOR CUSTOMER DETAILS (New Logic)
        cust_lookup = df[['Partner Customer Code', 'Partner Customer Name', 'Salesman Name', 'Partner Customer District']].drop_duplicates('Partner Customer Code')
        cust_lookup['Partner Customer Code'] = cust_lookup['Partner Customer Code'].astype(str)
        latest['customer_id'] = latest['customer_id'].astype(str)

        results = pd.DataFrame({
            'customer_id': latest['customer_id'].values,
            'product_id': latest['product_id'].values,
            'reorder_probability': reorder_probs,
            'predicted_quantity': np.maximum(0, np.round(quantities)).astype(int),
            'days_since_last_order': latest['days_since_last_order'].values,
            'avg_discount': latest.get('avg_discount', [0]*len(latest)).values
        })

        # 2. MERGE DETAILS
        results = results.merge(
            cust_lookup,
            left_on='customer_id',
            right_on='Partner Customer Code',
            how='left'
        )

        results['Partner Customer Name'] = results['Partner Customer Name'].fillna('Unknown Customer')
        results['Salesman Name'] = results['Salesman Name'].fillna('N/A')

        results['priority_score'] = (
            results['reorder_probability'] * 0.7 +
            (results['predicted_quantity'] / (results['predicted_quantity'].max() + 1)) * 0.3
        )

        results = results.sort_values('priority_score', ascending=False)
        return results.head(top_k)

    def get_model_comparison(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from all models for comparison
        """
        predictions = {}

        # Reorder likelihood
        predictions['reorder'] = {}
        for model_name in ['ffnn', 'lgbm']:
            if self.reorder_models[model_name] is not None:
                try:
                    predictions['reorder'][model_name] = self.predict_reorder_likelihood(X, model_name)
                except:
                    pass
        predictions['reorder']['ensemble'] = self.predict_reorder_likelihood(X, 'ensemble')

        # Quantity
        predictions['quantity'] = {}
        for model_name in ['ffnn', 'lgbm']:
            if self.quantity_models[model_name] is not None:
                try:
                    predictions['quantity'][model_name] = self.predict_quantity(X, model_name)
                except:
                    pass
        predictions['quantity']['ensemble'] = self.predict_quantity(X, 'ensemble')

        return predictions
