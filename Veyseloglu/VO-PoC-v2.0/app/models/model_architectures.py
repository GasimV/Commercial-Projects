"""
Model Architectures: FFNN, LSTM, and LightGBM
For reorder likelihood and quantity prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import joblib
import os
import json

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

# Traditional ML
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)

import warnings
warnings.filterwarnings('ignore')


class FFNNModel:
    """
    Feed-Forward Neural Network for both classification and regression
    """

    def __init__(self, input_dim: int, task: str = 'classification'):
        self.input_dim = input_dim
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def build_model(self):
        """Build FFNN architecture"""
        inputs = layers.Input(shape=(self.input_dim,))

        # Deep architecture with batch normalization and dropout
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Output layer
        if self.task == 'classification':
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        else:  # regression
            outputs = layers.Dense(1, activation='relu')(x)
            loss = 'huber'  # Robust to outliers
            metrics = ['mae', 'mse']

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )

        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 256) -> Dict:
        """Train the model"""

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Build model if not already built
        if self.model is None:
            self.build_model()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=0.00001,
                verbose=1
            )
        ]

        # Train
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        metrics = self.evaluate(X_val_scaled, y_val)

        return metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        y_pred = self.model.predict(X, verbose=0).flatten()

        if self.task == 'classification':
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics = {
                'roc_auc': float(roc_auc_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred_binary)),
                'precision': float(precision_score(y, y_pred_binary)),
                'recall': float(recall_score(y, y_pred_binary))
            }
        else:
            metrics = {
                'mae': float(mean_absolute_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'r2': float(r2_score(y, y_pred))
            }

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

    def save(self, path: str):
        """Save model and scaler"""
        self.model.save(f"{path}_model.h5")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")

        # Save metrics
        if self.history:
            history_dict = {k: [float(v) for v in val]
                          for k, val in self.history.history.items()}
            with open(f"{path}_history.json", 'w') as f:
                json.dump(history_dict, f)

    def load(self, path: str):
        """Load model and scaler"""
        self.model = keras.models.load_model(f"{path}_model.h5")
        self.scaler = joblib.load(f"{path}_scaler.pkl")


class LSTMModel:
    """
    LSTM Neural Network for sequence-based prediction
    """

    def __init__(self, input_shape: Tuple[int, int], task: str = 'classification'):
        self.input_shape = input_shape  # (sequence_length, n_features)
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def build_model(self):
        """Build LSTM architecture"""
        inputs = layers.Input(shape=self.input_shape)

        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
        x = layers.Dropout(0.3)(x)

        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
        x = layers.Dropout(0.2)(x)

        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Output layer
        if self.task == 'classification':
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
        else:
            outputs = layers.Dense(1, activation='relu')(x)
            loss = 'huber'
            metrics = ['mae', 'mse']

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )

        return self.model

    def prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """Scale sequences properly"""
        n_samples, seq_len, n_features = X.shape

        # Reshape for scaling
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)

        # Reshape back
        return X_scaled.reshape(n_samples, seq_len, n_features)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 128) -> Dict:
        """Train the model"""

        # Scale sequences
        X_train_scaled = self.prepare_sequences(X_train)

        # Scale validation using same scaler
        n_samples, seq_len, n_features = X_val.shape
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(n_samples, seq_len, n_features)

        # Build model
        if self.model is None:
            self.build_model()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
        ]

        # Train
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        metrics = self.evaluate(X_val_scaled, y_val)

        return metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        y_pred = self.model.predict(X, verbose=0).flatten()

        if self.task == 'classification':
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics = {
                'roc_auc': float(roc_auc_score(y, y_pred)),
                'f1': float(f1_score(y, y_pred_binary)),
                'precision': float(precision_score(y, y_pred_binary)),
                'recall': float(recall_score(y, y_pred_binary))
            }
        else:
            metrics = {
                'mae': float(mean_absolute_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'r2': float(r2_score(y, y_pred))
            }

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        # Scale sequences
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)

        return self.model.predict(X_scaled, verbose=0).flatten()

    def save(self, path: str):
        """Save model and scaler"""
        self.model.save(f"{path}_model.h5")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")

        if self.history:
            history_dict = {k: [float(v) for v in val]
                          for k, val in self.history.history.items()}
            with open(f"{path}_history.json", 'w') as f:
                json.dump(history_dict, f)

    def load(self, path: str):
        """Load model and scaler"""
        self.model = keras.models.load_model(f"{path}_model.h5")
        self.scaler = joblib.load(f"{path}_scaler.pkl")


class LightGBMModel:
    """
    LightGBM for both classification and regression
    """

    def __init__(self, task: str = 'classification'):
        self.task = task
        self.model = None
        self.scaler = StandardScaler()

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train LightGBM model"""

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Create datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

        # Parameters
        if self.task == 'classification':
            params = {
                'objective': 'binary',
                'metric': ['auc', 'binary_logloss'],
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 7,
                'min_child_samples': 20
            }
        else:
            params = {
                'objective': 'regression',
                'metric': ['l1', 'l2'],
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'max_depth': 7,
                'min_child_samples': 20
            }

        # Train
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        # Evaluate
        metrics = self.evaluate(X_val_scaled, y_val)

        return metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if self.task == 'classification':
            y_pred_proba = self.model.predict(X)
            y_pred_binary = (y_pred_proba > 0.5).astype(int)

            metrics = {
                'roc_auc': float(roc_auc_score(y, y_pred_proba)),
                'f1': float(f1_score(y, y_pred_binary)),
                'precision': float(precision_score(y, y_pred_binary)),
                'recall': float(recall_score(y, y_pred_binary))
            }
        else:
            y_pred = self.model.predict(X)
            metrics = {
                'mae': float(mean_absolute_error(y, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
                'r2': float(r2_score(y, y_pred))
            }

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        importance = self.model.feature_importance(importance_type='gain')
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

    def save(self, path: str):
        """Save model and scaler"""
        self.model.save_model(f"{path}_model.txt")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")

    def load(self, path: str):
        """Load model and scaler"""
        self.model = lgb.Booster(model_file=f"{path}_model.txt")
        self.scaler = joblib.load(f"{path}_scaler.pkl")


class EnsembleModel:
    """
    Ensemble of FFNN, LSTM, and LightGBM
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {'ffnn': 0.33, 'lstm': 0.33, 'lgbm': 0.34}
        self.models = {}

    def add_model(self, name: str, model: Any):
        """Add a model to the ensemble"""
        self.models[name] = model

    def predict(self, X_tabular: np.ndarray, X_sequence: np.ndarray = None) -> np.ndarray:
        """Make ensemble predictions with basic shape checking."""
        predictions = []
        base_len = None

        def add_pred(name: str, arr: np.ndarray, weight: float):
            nonlocal base_len
            arr = np.asarray(arr).flatten()
            if base_len is None:
                base_len = arr.shape[0]
            if arr.shape[0] != base_len:
                # Length mismatch – skip this model for the ensemble
                print(
                    f"[Ensemble] Skipping '{name}' due to length mismatch: "
                    f"expected {base_len}, got {arr.shape[0]}"
                )
                return
            predictions.append(arr * weight)

        # FFNN (tabular)
        if 'ffnn' in self.models and self.models['ffnn'] is not None:
            pred = self.models['ffnn'].predict(X_tabular)
            add_pred('ffnn', pred, self.weights.get('ffnn', 0.0))

        # LightGBM (tabular)
        if 'lgbm' in self.models and self.models['lgbm'] is not None:
            pred = self.models['lgbm'].predict(X_tabular)
            add_pred('lgbm', pred, self.weights.get('lgbm', 0.0))

        # LSTM (sequence)
        if 'lstm' in self.models and self.models['lstm'] is not None and X_sequence is not None:
            pred = self.models['lstm'].predict(X_sequence)
            add_pred('lstm', pred, self.weights.get('lstm', 0.0))

        if not predictions:
            raise ValueError("Ensemble has no valid predictions to combine.")

        return np.sum(predictions, axis=0)

    def save(self, path: str):
        """Save ensemble"""
        joblib.dump(self.weights, f"{path}_ensemble_weights.pkl")
