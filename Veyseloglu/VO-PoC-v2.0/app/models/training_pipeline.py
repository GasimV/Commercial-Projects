"""
Training Pipeline for Reorder Likelihood and Quantity Prediction
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split

from app.utils.feature_engineering import FeatureEngineer
from app.models.model_architectures import (
    FFNNModel, LSTMModel, LightGBMModel, EnsembleModel
)


def time_based_split(df: pd.DataFrame, test_size: float = 0.2,
                     val_size: float = 0.2, gap_days: int = 0,
                     date_column: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform time-based train/val/test split to avoid data leakage in forecasting.

    Args:
        df: DataFrame with a date column
        test_size: Fraction of data for test set (most recent)
        val_size: Fraction of remaining data for validation
        gap_days: Optional gap between train and test to prevent label leakage
        date_column: Name of the date column

    Returns:
        train_df, val_df, test_df
    """
    # Sort by date
    df_sorted = df.sort_values(date_column).reset_index(drop=True)

    n = len(df_sorted)
    test_start_idx = int(n * (1 - test_size))

    # Apply gap if specified
    if gap_days > 0:
        test_start_date = df_sorted.iloc[test_start_idx][date_column]
        gap_start_date = test_start_date - pd.Timedelta(days=gap_days)
        # Remove rows in the gap period
        df_no_gap = df_sorted[~((df_sorted[date_column] >= gap_start_date) &
                                 (df_sorted[date_column] < test_start_date))].reset_index(drop=True)
        # Recalculate test start after gap removal
        n_no_gap = len(df_no_gap)
        test_start_idx = int(n_no_gap * (1 - test_size))
        df_sorted = df_no_gap

    # Split test
    test_df = df_sorted.iloc[test_start_idx:].reset_index(drop=True)
    remaining_df = df_sorted.iloc[:test_start_idx].reset_index(drop=True)

    # Split val from remaining
    n_remaining = len(remaining_df)
    val_start_idx = int(n_remaining * (1 - val_size))
    val_df = remaining_df.iloc[val_start_idx:].reset_index(drop=True)
    train_df = remaining_df.iloc[:val_start_idx].reset_index(drop=True)

    print(f"\nTime-based split (with {gap_days}-day gap):")
    print(f"  Train: {len(train_df):,} samples | {train_df[date_column].min()} to {train_df[date_column].max()}")
    print(f"  Val:   {len(val_df):,} samples | {val_df[date_column].min()} to {val_df[date_column].max()}")
    print(f"  Test:  {len(test_df):,} samples | {test_df[date_column].min()} to {test_df[date_column].max()}")

    return train_df, val_df, test_df


class ReorderTrainingPipeline:
    """
    Complete training pipeline for reorder likelihood prediction
    """

    def __init__(self, model_dir: str = 'models_store', data_dir: str = 'data', prediction_horizon: int = 30):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.prediction_horizon = prediction_horizon
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)

        self.engineer = FeatureEngineer(prediction_horizon=prediction_horizon)
        self.models = {
            'ffnn': None,
            'lstm': None,
            'lgbm': None,
            'ensemble': None
        }
        self.metrics = {}

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, resume_training: bool = False) -> Tuple:
        """
        Prepare data for training
        """
        print("=" * 80)
        print(f"REORDER LIKELIHOOD - DATA PREPARATION (Horizon: {self.prediction_horizon} days)")
        print("=" * 80)

        # Use horizon-aware feature cache path
        feature_file_path = os.path.join(self.data_dir, 'processed', f'features_full_h{self.prediction_horizon}.parquet')

        if resume_training and os.path.exists(feature_file_path):
            print(f"Resuming: Loading engineered features from {feature_file_path}")
            df_features = self.engineer.load_features(feature_file_path)
        else:
            print(f"Starting fresh feature engineering for {self.prediction_horizon}-day horizon...")
            # Build features
            df_features = self.engineer.build_features(df, create_targets=True)
            # Save features for future use
            self.engineer.save_features(df_features, feature_file_path)

        # Get feature columns
        feature_cols = self.engineer.get_feature_columns()['all']

        # Time-based split (critical for forecasting to avoid leakage)
        # Use gap equal to prediction horizon to prevent label leakage
        train_df, val_df, test_df = time_based_split(
            df_features,
            test_size=test_size,
            val_size=0.2,
            gap_days=self.prediction_horizon,
            date_column='date'
        )

        # Extract features and targets
        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df['will_reorder'].to_numpy(dtype=np.float32)

        X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
        y_val = val_df['will_reorder'].to_numpy(dtype=np.float32)

        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_df['will_reorder'].to_numpy(dtype=np.float32)

        print(f"\nSplit Statistics:")
        print(f"  Train: {X_train.shape}, Positive rate: {y_train.mean():.4f}")
        print(f"  Val:   {X_val.shape}, Positive rate: {y_val.mean():.4f}")
        print(f"  Test:  {X_test.shape}, Positive rate: {y_test.mean():.4f}")

        # Prepare LSTM sequences (with caching)
        seq_x_path = os.path.join(self.data_dir, 'processed', 'reorder_seq_X.npy')
        seq_y_path = os.path.join(self.data_dir, 'processed', 'reorder_seq_y.npy')

        if resume_training and os.path.exists(seq_x_path) and os.path.exists(seq_y_path):
            print("\nResuming: Loading LSTM sequences from disk...")
            # Allow object arrays in case sequence generation created ragged arrays
            X_seq = np.load(seq_x_path, allow_pickle=True)
            y_seq_like = np.load(seq_y_path, allow_pickle=True)
        else:
            print("\nPreparing LSTM sequences...")
            X_seq, y_seq_like, _ = self.engineer.prepare_sequences_for_lstm(
                df_features, sequence_length=10
            )
            print("Saving LSTM sequences to disk...")
            np.save(seq_x_path, X_seq)
            np.save(seq_y_path, y_seq_like)

        if len(X_seq) > 0:
            # For LSTM, we use the same time-based indices from the tabular split
            # This ensures LSTM sees the same temporal split as FFNN/LightGBM
            train_indices = train_df.index
            val_indices = val_df.index
            test_indices = test_df.index

            # Map sequence indices to split (sequences are already time-ordered)
            # Note: This is a simplification - for production you may want more sophisticated mapping
            n_seq = len(X_seq)
            seq_test_start = int(n_seq * (1 - test_size))
            seq_val_start = int(seq_test_start * 0.8)

            X_seq_train = X_seq[:seq_val_start]
            y_seq_train = y_seq_like[:seq_val_start]

            X_seq_val = X_seq[seq_val_start:seq_test_start]
            y_seq_val = y_seq_like[seq_val_start:seq_test_start]

            X_seq_test = X_seq[seq_test_start:]
            y_seq_test = y_seq_like[seq_test_start:]

            print(f"\nLSTM Sequences:")
            print(f"  Train: {X_seq_train.shape}")
            print(f"  Val:   {X_seq_val.shape}")
            print(f"  Test:  {X_seq_test.shape}")
        else:
            X_seq_train = X_seq_val = X_seq_test = None
            y_seq_train = y_seq_val = y_seq_test = None
            print("\nNot enough data for LSTM sequences")

        return {
            'tabular': {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            },
            'sequences': {
                'X_train': X_seq_train, 'y_train': y_seq_train,
                'X_val': X_seq_val, 'y_val': y_seq_val,
                'X_test': X_seq_test, 'y_test': y_seq_test
            },
            'feature_cols': feature_cols
        }

    def train_ffnn(self, data: Dict, resume_training: bool = False) -> Dict:
        """Train FFNN model"""
        tab = data['tabular']
        model_path = os.path.join(self.model_dir, 'reorder_likelihood_ffnn')
        full_path = f"{model_path}_model.h5"

        self.models['ffnn'] = FFNNModel(
            input_dim=tab['X_train'].shape[1],
            task='classification'
        )

        # Check existing
        if resume_training and os.path.exists(full_path):
            print(f"\n[RESUME] Found existing FFNN model at {full_path}. Loading...")
            try:
                self.models['ffnn'].load(model_path)
                test_metrics = self.models['ffnn'].evaluate(
                    self.models['ffnn'].scaler.transform(tab['X_test']),
                    tab['y_test']
                )
                print("✓ FFNN Loaded. Metrics:", test_metrics)
                self.metrics['ffnn'] = test_metrics
                return test_metrics
            except Exception as e:
                print(f"Failed to load model: {e}. Starting fresh training...")

        print("\n" + "=" * 80)
        print("TRAINING FFNN MODEL")
        print("=" * 80)

        metrics = self.models['ffnn'].train(
            tab['X_train'], tab['y_train'],
            tab['X_val'], tab['y_val'],
            epochs=100,
            batch_size=256
        )

        test_metrics = self.models['ffnn'].evaluate(
            self.models['ffnn'].scaler.transform(tab['X_test']),
            tab['y_test']
        )
        self.models['ffnn'].save(model_path)
        self.metrics['ffnn'] = test_metrics
        return test_metrics

    def train_lstm(self, data: Dict, resume_training: bool = False) -> Dict:
        """Train LSTM model"""
        seq = data['sequences']
        if seq['X_train'] is None:
            return {}

        model_path = os.path.join(self.model_dir, 'reorder_likelihood_lstm')
        full_path = f"{model_path}_model.h5"

        self.models['lstm'] = LSTMModel(
            input_shape=(seq['X_train'].shape[1], seq['X_train'].shape[2]),
            task='classification'
        )

        if resume_training and os.path.exists(full_path):
            print(f"\n[RESUME] Found existing LSTM model at {full_path}. Loading...")
            try:
                self.models['lstm'].load(model_path)

                n_samples, seq_len, n_features = seq['X_test'].shape
                X_test_reshaped = seq['X_test'].reshape(-1, n_features)
                X_test_scaled = self.models['lstm'].scaler.transform(X_test_reshaped)
                X_test_scaled = X_test_scaled.reshape(n_samples, seq_len, n_features)

                test_metrics = self.models['lstm'].evaluate(X_test_scaled, seq['y_test'])
                print("✓ LSTM Loaded. Metrics:", test_metrics)
                self.metrics['lstm'] = test_metrics
                return test_metrics
            except Exception as e:
                print(f"⚠ Failed to load model: {e}. Starting fresh training...")

        print("\n" + "=" * 80)
        print("TRAINING LSTM MODEL")
        print("=" * 80)

        metrics = self.models['lstm'].train(
            seq['X_train'], seq['y_train'],
            seq['X_val'], seq['y_val'],
            epochs=100,
            batch_size=128
        )

        n_samples, seq_len, n_features = seq['X_test'].shape
        X_test_reshaped = seq['X_test'].reshape(-1, n_features)
        X_test_scaled = self.models['lstm'].scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(n_samples, seq_len, n_features)

        test_metrics = self.models['lstm'].evaluate(X_test_scaled, seq['y_test'])
        self.models['lstm'].save(model_path)
        self.metrics['lstm'] = test_metrics
        return test_metrics

    def train_lgbm(self, data: Dict, resume_training: bool = False) -> Dict:
        """Train LightGBM model"""
        tab = data['tabular']
        model_path = os.path.join(self.model_dir, 'reorder_likelihood_lgbm')
        full_path = f"{model_path}_model.txt"

        self.models['lgbm'] = LightGBMModel(task='classification')

        if resume_training and os.path.exists(full_path):
            print(f"\n[RESUME] Found existing LightGBM model at {full_path}. Loading...")
            try:
                self.models['lgbm'].load(model_path)
                test_metrics = self.models['lgbm'].evaluate(
                    self.models['lgbm'].scaler.transform(tab['X_test']),
                    tab['y_test']
                )
                print("✓ LightGBM Loaded. Metrics:", test_metrics)
                self.metrics['lgbm'] = test_metrics
                return test_metrics
            except Exception as e:
                print(f"⚠ Failed to load model: {e}. Starting fresh training...")

        print("\n" + "=" * 80)
        print("TRAINING LIGHTGBM MODEL")
        print("=" * 80)

        metrics = self.models['lgbm'].train(
            tab['X_train'], tab['y_train'],
            tab['X_val'], tab['y_val']
        )

        test_metrics = self.models['lgbm'].evaluate(
            self.models['lgbm'].scaler.transform(tab['X_test']),
            tab['y_test']
        )

        self.models['lgbm'].save(model_path)
        importance = self.models['lgbm'].get_feature_importance()
        with open(os.path.join(self.model_dir, 'reorder_feature_importance.json'), 'w') as f:
            json.dump(importance, f, indent=2)

        self.metrics['lgbm'] = test_metrics
        return test_metrics

    def create_ensemble(self, data: Dict) -> Dict:
        """Create ensemble model (FFNN + LightGBM only for production inference)"""
        print("\n" + "=" * 80)
        print("CREATING ENSEMBLE MODEL (FFNN + LightGBM)")
        print("=" * 80)
        print("Note: LSTM trained but excluded from ensemble for tabular inference")

        tab = data['tabular']
        seq = data['sequences']

        # Ensemble uses FFNN + LightGBM only (50/50 weights)
        # LSTM excluded because it requires sequences at inference time
        self.models['ensemble'] = EnsembleModel(
            weights={'ffnn': 0.5, 'lstm': 0.0, 'lgbm': 0.5}
        )
        self.models['ensemble'].add_model('ffnn', self.models['ffnn'])
        self.models['ensemble'].add_model('lstm', self.models['lstm'])
        self.models['ensemble'].add_model('lgbm', self.models['lgbm'])

        # Always use tabular-only prediction for ensemble
        y_pred = self.models['ensemble'].predict(tab['X_test'])

        from sklearn.metrics import roc_auc_score, f1_score
        y_pred_binary = (y_pred > 0.5).astype(int)
        ensemble_metrics = {
            'roc_auc': float(roc_auc_score(tab['y_test'], y_pred)),
            'f1': float(f1_score(tab['y_test'], y_pred_binary))
        }

        print("\n--- Ensemble Test Metrics ---")
        for key, val in ensemble_metrics.items():
            print(f"{key}: {val:.4f}")

        self.models['ensemble'].save(os.path.join(self.model_dir, 'reorder_likelihood_ensemble'))
        self.metrics['ensemble'] = ensemble_metrics
        return ensemble_metrics

    def train_all(self, df: pd.DataFrame, resume_training: bool = False) -> Dict:
        """Train all models"""
        data = self.prepare_data(df, resume_training=resume_training)

        self.train_ffnn(data, resume_training=resume_training)
        self.train_lstm(data, resume_training=resume_training)
        self.train_lgbm(data, resume_training=resume_training)
        self.create_ensemble(data)

        with open(os.path.join(self.model_dir, 'reorder_likelihood_metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)

        return self.metrics


class QuantityTrainingPipeline:
    """
    Complete training pipeline for quantity prediction
    """

    def __init__(self, model_dir: str = 'models_store', data_dir: str = 'data', prediction_horizon: int = 30):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.prediction_horizon = prediction_horizon
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)

        self.engineer = FeatureEngineer(prediction_horizon=prediction_horizon)
        self.models = {
            'ffnn': None,
            'lstm': None,
            'lgbm': None,
            'ensemble': None
        }
        self.metrics = {}

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2, resume_training: bool = False) -> Tuple:
        """
        Prepare data for training
        """
        print("=" * 80)
        print(f"QUANTITY PREDICTION - DATA PREPARATION (Horizon: {self.prediction_horizon} days)")
        print("=" * 80)

        # Use horizon-aware feature cache path
        feature_file_path = os.path.join(self.data_dir, 'processed', f'features_full_h{self.prediction_horizon}.parquet')

        if resume_training and os.path.exists(feature_file_path):
            print(f"Resuming: Loading engineered features from {feature_file_path}")
            df_features = self.engineer.load_features(feature_file_path)
        else:
            if not 'df_features' in locals():
                 print(f"Starting fresh feature engineering for {self.prediction_horizon}-day horizon (Quantity)...")
                 df_features = self.engineer.build_features(df, create_targets=True)
                 self.engineer.save_features(df_features, feature_file_path)

        feature_cols = self.engineer.get_feature_columns()['all']

        # Filter to only rows with valid quantity targets (within horizon)
        df_qty = df_features[df_features['next_order_quantity'].notna()].copy()

        # Time-based split (same as reorder pipeline)
        train_df, val_df, test_df = time_based_split(
            df_qty,
            test_size=test_size,
            val_size=0.2,
            gap_days=self.prediction_horizon,
            date_column='date'
        )

        # Extract features and targets
        X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_train = train_df['next_order_quantity'].to_numpy(dtype=np.float32)

        X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
        y_val = val_df['next_order_quantity'].to_numpy(dtype=np.float32)

        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        y_test = test_df['next_order_quantity'].to_numpy(dtype=np.float32)

        print(f"\nSplit Statistics:")
        print(f"  Train: {X_train.shape}, Mean qty: {y_train.mean():.2f}")
        print(f"  Val:   {X_val.shape}, Mean qty: {y_val.mean():.2f}")
        print(f"  Test:  {X_test.shape}, Mean qty: {y_test.mean():.2f}")

        seq_x_path = os.path.join(self.data_dir, 'processed', 'qty_seq_X.npy')
        seq_y_path = os.path.join(self.data_dir, 'processed', 'qty_seq_y.npy')

        if resume_training and os.path.exists(seq_x_path) and os.path.exists(seq_y_path):
            print("\nResuming: Loading LSTM sequences from disk...")
            X_seq = np.load(seq_x_path, allow_pickle=True)
            y_seq_qty = np.load(seq_y_path, allow_pickle=True)
        else:
            print("\nPreparing LSTM sequences...")
            X_seq, _, y_seq_qty = self.engineer.prepare_sequences_for_lstm(
                df_features, sequence_length=10
            )
            valid_mask = ~np.isnan(y_seq_qty)
            X_seq = X_seq[valid_mask]
            y_seq_qty = y_seq_qty[valid_mask]
            print("Saving LSTM sequences to disk...")
            np.save(seq_x_path, X_seq)
            np.save(seq_y_path, y_seq_qty)

        if len(X_seq) > 0:
            # Use time-based split for LSTM sequences (same as tabular)
            n_seq = len(X_seq)
            seq_test_start = int(n_seq * (1 - test_size))
            seq_val_start = int(seq_test_start * 0.8)

            X_seq_train = X_seq[:seq_val_start]
            y_seq_train = y_seq_qty[:seq_val_start]

            X_seq_val = X_seq[seq_val_start:seq_test_start]
            y_seq_val = y_seq_qty[seq_val_start:seq_test_start]

            X_seq_test = X_seq[seq_test_start:]
            y_seq_test = y_seq_qty[seq_test_start:]

            print(f"\nLSTM Sequences:")
            print(f"  Train: {X_seq_train.shape}")
            print(f"  Val:   {X_seq_val.shape}")
            print(f"  Test:  {X_seq_test.shape}")
        else:
            X_seq_train = X_seq_val = X_seq_test = None
            y_seq_train = y_seq_val = y_seq_test = None
            print("\n⚠ Not enough data for LSTM sequences")

        return {
            'tabular': {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            },
            'sequences': {
                'X_train': X_seq_train, 'y_train': y_seq_train,
                'X_val': X_seq_val, 'y_val': y_seq_val,
                'X_test': X_seq_test, 'y_test': y_seq_test
            },
            'feature_cols': feature_cols
        }

    def train_ffnn(self, data: Dict, resume_training: bool = False) -> Dict:
        """Train FFNN model"""
        tab = data['tabular']
        model_path = os.path.join(self.model_dir, 'quantity_prediction_ffnn')
        full_path = f"{model_path}_model.h5"

        self.models['ffnn'] = FFNNModel(
            input_dim=tab['X_train'].shape[1],
            task='regression'
        )

        if resume_training and os.path.exists(full_path):
            print(f"\n[RESUME] Found existing FFNN model at {full_path}. Loading...")
            try:
                self.models['ffnn'].load(model_path)
                test_metrics = self.models['ffnn'].evaluate(
                    self.models['ffnn'].scaler.transform(tab['X_test']),
                    tab['y_test']
                )
                print("✓ FFNN Loaded. Metrics:", test_metrics)
                self.metrics['ffnn'] = test_metrics
                return test_metrics
            except Exception as e:
                print(f"⚠ Failed to load model: {e}. Starting fresh training...")

        print("\n" + "=" * 80)
        print("TRAINING FFNN MODEL")
        print("=" * 80)

        metrics = self.models['ffnn'].train(
            tab['X_train'], tab['y_train'],
            tab['X_val'], tab['y_val'],
            epochs=100,
            batch_size=256
        )

        test_metrics = self.models['ffnn'].evaluate(
            self.models['ffnn'].scaler.transform(tab['X_test']),
            tab['y_test']
        )
        self.models['ffnn'].save(model_path)
        self.metrics['ffnn'] = test_metrics
        return test_metrics

    def train_lstm(self, data: Dict, resume_training: bool = False) -> Dict:
        """Train LSTM model"""
        seq = data['sequences']
        if seq['X_train'] is None:
            return {}

        model_path = os.path.join(self.model_dir, 'quantity_prediction_lstm')
        full_path = f"{model_path}_model.h5"

        self.models['lstm'] = LSTMModel(
            input_shape=(seq['X_train'].shape[1], seq['X_train'].shape[2]),
            task='regression'
        )

        if resume_training and os.path.exists(full_path):
            print(f"\n[RESUME] Found existing LSTM model at {full_path}. Loading...")
            try:
                self.models['lstm'].load(model_path)
                n_samples, seq_len, n_features = seq['X_test'].shape
                X_test_reshaped = seq['X_test'].reshape(-1, n_features)
                X_test_scaled = self.models['lstm'].scaler.transform(X_test_reshaped)
                X_test_scaled = X_test_scaled.reshape(n_samples, seq_len, n_features)
                test_metrics = self.models['lstm'].evaluate(X_test_scaled, seq['y_test'])
                print("✓ LSTM Loaded. Metrics:", test_metrics)
                self.metrics['lstm'] = test_metrics
                return test_metrics
            except Exception as e:
                print(f"⚠ Failed to load model: {e}. Starting fresh training...")

        print("\n" + "=" * 80)
        print("TRAINING LSTM MODEL")
        print("=" * 80)

        metrics = self.models['lstm'].train(
            seq['X_train'], seq['y_train'],
            seq['X_val'], seq['y_val'],
            epochs=100,
            batch_size=128
        )

        n_samples, seq_len, n_features = seq['X_test'].shape
        X_test_reshaped = seq['X_test'].reshape(-1, n_features)
        X_test_scaled = self.models['lstm'].scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(n_samples, seq_len, n_features)

        test_metrics = self.models['lstm'].evaluate(X_test_scaled, seq['y_test'])
        self.models['lstm'].save(model_path)
        self.metrics['lstm'] = test_metrics
        return test_metrics

    def train_lgbm(self, data: Dict, resume_training: bool = False) -> Dict:
        """Train LightGBM model"""
        tab = data['tabular']
        model_path = os.path.join(self.model_dir, 'quantity_prediction_lgbm')
        full_path = f"{model_path}_model.txt"

        self.models['lgbm'] = LightGBMModel(task='regression')

        if resume_training and os.path.exists(full_path):
            print(f"\n[RESUME] Found existing LightGBM model at {full_path}. Loading...")
            try:
                self.models['lgbm'].load(model_path)
                test_metrics = self.models['lgbm'].evaluate(
                    self.models['lgbm'].scaler.transform(tab['X_test']),
                    tab['y_test']
                )
                print("✓ LightGBM Loaded. Metrics:", test_metrics)
                self.metrics['lgbm'] = test_metrics
                return test_metrics
            except Exception as e:
                print(f"⚠ Failed to load model: {e}. Starting fresh training...")

        print("\n" + "=" * 80)
        print("TRAINING LIGHTGBM MODEL")
        print("=" * 80)

        metrics = self.models['lgbm'].train(
            tab['X_train'], tab['y_train'],
            tab['X_val'], tab['y_val']
        )

        test_metrics = self.models['lgbm'].evaluate(
            self.models['lgbm'].scaler.transform(tab['X_test']),
            tab['y_test']
        )
        self.models['lgbm'].save(model_path)
        importance = self.models['lgbm'].get_feature_importance()
        with open(os.path.join(self.model_dir, 'quantity_feature_importance.json'), 'w') as f:
            json.dump(importance, f, indent=2)

        self.metrics['lgbm'] = test_metrics
        return test_metrics

    def train_all(self, df: pd.DataFrame, resume_training: bool = False) -> Dict:
        """Train all models"""
        data = self.prepare_data(df, resume_training=resume_training)

        self.train_ffnn(data, resume_training=resume_training)
        self.train_lstm(data, resume_training=resume_training)
        self.train_lgbm(data, resume_training=resume_training)

        with open(os.path.join(self.model_dir, 'quantity_prediction_metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)

        return self.metrics