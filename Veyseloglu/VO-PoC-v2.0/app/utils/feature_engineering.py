"""
Advanced Feature Engineering for Reorder & Quantity Prediction
Implements comprehensive feature extraction from real sales data
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datetime import timedelta
import os
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for reorder likelihood and quantity prediction
    """

    def __init__(self, lookback_days: int = 90, prediction_horizon: int = 30):
        self.lookback_days = lookback_days
        self.prediction_horizon = prediction_horizon

    def save_features(self, df: pd.DataFrame, path: str):
        """
        Save engineered features to disk (Parquet format recommended)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert object columns to string for Parquet compatibility if needed
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)

        df.to_parquet(path, index=False)
        print(f"Features saved to {path}")

    def load_features(self, path: str) -> pd.DataFrame:
        """
        Load engineered features from disk
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Feature file not found at {path}")

        print(f"Loading features from {path}...")
        return pd.read_parquet(path)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean the raw sales data
        """
        df = df.copy()

        # Rename columns for easier handling
        column_mapping = {
            'DATE': 'date',
            'Partner Customer Referans Code': 'customer_id',
            'Product Code': 'product_id',
            'NetSalesQty': 'quantity',
            'Net Sales Value LC': 'sales_value',
            'Discount %': 'discount_pct',
            'Partner Customer Settlement': 'settlement',
            'Partner Customer District': 'district',
            'Product Level H1': 'category_h1',
            'Product Level H2': 'category_h2',
            'Product Level H3': 'category_h3',
            'Product Manufacturer': 'manufacturer',
            'Customer Channel Current Level 1 Manager': 'channel_l1',
            'Salesman Code': 'salesman_id'
        }

        df = df.rename(columns=column_mapping)

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Handle missing values
        df['discount_pct'] = df['discount_pct'].fillna(0)
        df['quantity'] = df['quantity'].fillna(0)
        df['sales_value'] = df['sales_value'].fillna(0)

        # Remove invalid records
        df = df[df['quantity'] > 0]
        df = df[df['customer_id'].notna()]
        df = df[df['product_id'].notna()]

        return df

    def create_recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create recency-based features (time since last order)
        """
        df = df.sort_values(['customer_id', 'product_id', 'date'])

        # Days since last order for this customer-product pair
        df['days_since_last_order'] = df.groupby(['customer_id', 'product_id'])['date'].diff().dt.days

        # Days since first order (customer tenure per product)
        df['days_since_first_order'] = df.groupby(['customer_id', 'product_id'])['date'].transform(
            lambda x: (x - x.min()).dt.days
        )

        # Fill NaN for first orders - cap at 3x prediction horizon to avoid extreme outliers
        # This prevents the model from seeing unrealistic values for first-time customers
        max_days_since_last = self.prediction_horizon * 3
        df['days_since_last_order'] = df['days_since_last_order'].fillna(max_days_since_last)

        return df

    def create_frequency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create frequency-based features (order patterns).
        For each customer–product pair we count how many *previous* orders
        occurred within horizon-relative lookback windows.
        """
        df = df.sort_values(['customer_id', 'product_id', 'date'])

        # Cumulative order count per customer-product
        df['order_count'] = df.groupby(['customer_id', 'product_id']).cumcount() + 1

        # Make sure date is datetime (should already be from prepare_data)
        df['date'] = pd.to_datetime(df['date'])

        # Sliding time-window counts per customer-product
        # Use horizon-adaptive windows: 1x, 2x, 3x the prediction horizon
        window_multipliers = [1, 2, 3]
        window_days = [self.prediction_horizon * mult for mult in window_multipliers]

        for days in window_days:
            col = f'orders_last_{days}d'
            df[col] = 0  # initialise
            window = pd.Timedelta(days=days)

            # Process each (customer, product) group separately
            for (_, _), idx in df.groupby(['customer_id', 'product_id']).groups.items():
                g = df.loc[idx].sort_values('date')
                dates = g['date'].values
                n = len(g)

                # Two-pointer sliding window; O(n) per group
                counts = np.zeros(n, dtype=np.int32)
                j = 0
                for i in range(n):
                    # Move left pointer until window is within `days`
                    while j < i and (dates[i] - dates[j]) > window:
                        j += 1
                    # Number of previous orders in the window
                    counts[i] = i - j

                df.loc[g.index, col] = counts

        # Average order interval (frequency)
        df['avg_order_interval'] = df.groupby(
            ['customer_id', 'product_id']
        )['days_since_last_order'].transform(lambda x: x.expanding().mean())

        # Order regularity (std of intervals - lower is more regular)
        df['order_interval_std'] = df.groupby(
            ['customer_id', 'product_id']
        )['days_since_last_order'].transform(lambda x: x.expanding().std())
        df['order_interval_std'] = df['order_interval_std'].fillna(0)

        return df

    def create_monetary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create monetary/value-based features
        """
        df = df.sort_values(['customer_id', 'product_id', 'date'])

        # Cumulative quantities and values
        df['cumulative_quantity'] = df.groupby(['customer_id', 'product_id'])['quantity'].cumsum()
        df['cumulative_value'] = df.groupby(['customer_id', 'product_id'])['sales_value'].cumsum()

        # Rolling averages (quantity)
        for window in [3, 5, 10]:
            df[f'qty_rolling_mean_{window}'] = df.groupby(['customer_id', 'product_id'])['quantity'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'qty_rolling_std_{window}'] = df.groupby(['customer_id', 'product_id'])['quantity'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )

        # Rolling averages (value)
        for window in [3, 5]:
            df[f'value_rolling_mean_{window}'] = df.groupby(['customer_id', 'product_id'])['sales_value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

        # Average unit price
        df['avg_unit_price'] = df['sales_value'] / (df['quantity'] + 0.01)

        # Discount features
        df['avg_discount'] = df.groupby(['customer_id', 'product_id'])['discount_pct'].transform(
            lambda x: x.expanding().mean()
        )

        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features (seasonality, day of week, etc.)
        """
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week

        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Is month start/end
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)

        # Cyclical encoding for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from categorical variables
        """
        # Customer-level aggregations
        df['customer_total_products'] = df.groupby('customer_id')['product_id'].transform('nunique')
        df['customer_total_orders'] = df.groupby('customer_id')['date'].transform('count')

        # Product-level aggregations
        df['product_total_customers'] = df.groupby('product_id')['customer_id'].transform('nunique')
        df['product_popularity'] = df.groupby('product_id')['quantity'].transform('sum')

        # Category performance
        df['category_h1_volume'] = df.groupby('category_h1')['quantity'].transform('sum')
        df['manufacturer_volume'] = df.groupby('manufacturer')['quantity'].transform('sum')

        # Geographic features
        df['settlement_customer_count'] = df.groupby('settlement')['customer_id'].transform('nunique')
        df['district_order_count'] = df.groupby('district')['date'].transform('count')

        return df

    def create_customer_product_interaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between customer and product
        """
        # Share of wallet: what % of customer's spend is on this product
        customer_total_value = df.groupby(['customer_id', 'date'])['sales_value'].transform('sum')
        df['share_of_wallet'] = df['sales_value'] / (customer_total_value + 0.01)

        # Product concentration: how diverse is customer's purchase
        df['customer_product_concentration'] = df.groupby('customer_id')['product_id'].transform(
            lambda x: 1.0 / x.nunique()
        )

        # Relative quantity: how much does this customer buy vs average
        product_avg_qty = df.groupby('product_id')['quantity'].transform('mean')
        df['relative_quantity'] = df['quantity'] / (product_avg_qty + 0.01)

        return df

    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend-based features (growth, momentum)
        """
        df = df.sort_values(['customer_id', 'product_id', 'date'])

        # Quantity trend (comparing recent vs older periods)
        df['qty_trend'] = df.groupby(['customer_id', 'product_id'])['quantity'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean() / (x.rolling(window=6, min_periods=1).mean() + 0.01)
        )

        # Purchase momentum (orders getting more frequent?)
        df['momentum'] = df.groupby(['customer_id', 'product_id'])['days_since_last_order'].transform(
            lambda x: x.shift(1) / (x + 0.01)
        )

        return df

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['customer_id', 'product_id', 'date'])

        df['next_order_date'] = df.groupby(['customer_id', 'product_id'])['date'].shift(-1)
        df['days_to_next_order'] = (df['next_order_date'] - df['date']).dt.days

        # Default: will_reorder based on next order timing (works when next_order_date exists)
        df['will_reorder'] = (df['days_to_next_order'] <= self.prediction_horizon).astype(int)

        df['next_order_quantity'] = df.groupby(['customer_id', 'product_id'])['quantity'].shift(-1)
        df.loc[df['days_to_next_order'] > self.prediction_horizon, 'next_order_quantity'] = np.nan

        # ---- FIX: right-censoring instead of dropping all last orders ----
        dataset_end = df['date'].max()
        df['days_until_dataset_end'] = (dataset_end - df['date']).dt.days

        missing_next = df['next_order_date'].isna()

        # If there is no next order, but we observed at least horizon days after this row,
        # then it's a true negative (no reorder within horizon).
        df.loc[missing_next & (df['days_until_dataset_end'] >= self.prediction_horizon), 'will_reorder'] = 0
        df.loc[missing_next & (df['days_until_dataset_end'] >= self.prediction_horizon), 'next_order_quantity'] = np.nan

        # If there is no next order AND we did NOT observe a full horizon after this row,
        # the label is censored/unknown -> drop.
        censored = missing_next & (df['days_until_dataset_end'] < self.prediction_horizon)

        df_with_target = df.loc[~censored].copy()
        return df_with_target


    def get_feature_columns(self) -> Dict[str, List[str]]:
        """
        Return lists of feature columns by category
        """
        recency_features = [
            'days_since_last_order', 'days_since_first_order'
        ]

        # Dynamically generate frequency feature names based on horizon
        window_days = [self.prediction_horizon * mult for mult in [1, 2, 3]]
        frequency_features = [
            'order_count',
            *[f'orders_last_{days}d' for days in window_days],
            'avg_order_interval', 'order_interval_std'
        ]

        monetary_features = [
            'cumulative_quantity', 'cumulative_value',
            'qty_rolling_mean_3', 'qty_rolling_mean_5', 'qty_rolling_mean_10',
            'qty_rolling_std_3', 'qty_rolling_std_5', 'qty_rolling_std_10',
            'value_rolling_mean_3', 'value_rolling_mean_5',
            'avg_unit_price', 'avg_discount'
        ]

        temporal_features = [
            'month', 'quarter', 'day_of_week', 'is_weekend',
            'is_month_start', 'is_month_end',
            'month_sin', 'month_cos', 'dow_sin', 'dow_cos'
        ]

        # categorical_features = [
        #     'customer_total_products', 'customer_total_orders',
        #     'product_total_customers', 'product_popularity',
        #     'category_h1_volume', 'manufacturer_volume',
        #     'settlement_customer_count', 'district_order_count'
        # ]

        # interaction_features = [
        #     'share_of_wallet', 'customer_product_concentration', 'relative_quantity'
        # ]

        trend_features = [
            'qty_trend', 'momentum'
        ]

        all_features = (recency_features + frequency_features + monetary_features +
                        temporal_features + trend_features) # categorical_features + interaction_features +

        return {
            'all': all_features,
            'recency': recency_features,
            'frequency': frequency_features,
            'monetary': monetary_features,
            'temporal': temporal_features,
            #'categorical': categorical_features,
            #'interaction': interaction_features,
            'trend': trend_features
        }

    def build_features(self, df: pd.DataFrame, create_targets: bool = True) -> pd.DataFrame:
        """
        Build all features from raw data
        """
        print("Preparing data...")
        df = self.prepare_data(df)

        print("Creating recency features...")
        df = self.create_recency_features(df)

        print("Creating frequency features...")
        df = self.create_frequency_features(df)

        print("Creating monetary features...")
        df = self.create_monetary_features(df)

        print("Creating temporal features...")
        df = self.create_temporal_features(df)

        # print("Creating categorical features...")
        # df = self.create_categorical_features(df)

        # print("Creating interaction features...")
        # df = self.create_customer_product_interaction(df)

        print("Creating trend features...")
        df = self.create_trend_features(df)

        if create_targets:
            print("Creating target variables...")
            df = self.create_target_variable(df)

        # Fill remaining NaN values
        feature_cols = self.get_feature_columns()['all']
        df[feature_cols] = df[feature_cols].fillna(0)

        print(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def prepare_sequences_for_lstm(self, df: pd.DataFrame,
                                   sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM models
        """
        feature_cols = self.get_feature_columns()['all']

        sequences = []
        targets_likelihood = []
        targets_quantity = []

        # Group by customer-product and create sequences
        for (customer, product), group in df.groupby(['customer_id', 'product_id']):
            if len(group) < sequence_length + 1:
                continue

            group = group.sort_values('date')

            for i in range(len(group) - sequence_length):
                seq = group.iloc[i:i + sequence_length][feature_cols].values
                target_like = group.iloc[i + sequence_length]['will_reorder']
                target_qty = group.iloc[i + sequence_length]['next_order_quantity']

                sequences.append(seq)
                targets_likelihood.append(target_like)
                targets_quantity.append(target_qty)

        return (np.array(sequences),
                np.array(targets_likelihood),
                np.array(targets_quantity))


def get_latest_features_for_inference(df: pd.DataFrame,
                                      customer_id: str = None,
                                      product_id: str = None,
                                      prediction_horizon: int = 30) -> pd.DataFrame:
    """
    Get latest feature snapshot for inference

    Args:
        df: Raw sales data (must include enough historical data for rolling features!)
        customer_id: Optional customer filter
        product_id: Optional product filter
        prediction_horizon: Prediction horizon used during training (must match!)

    Important:
        - ALWAYS builds features from the provided df (new data)
        - The df should contain enough historical data to compute rolling features
        - Recommended: include at least 90 days of history before the prediction date
    """
    # ALWAYS build features from the provided data for inference
    # Do NOT load cached features - those are from training and would be stale
    print(f"Building features from uploaded data for inference (horizon: {prediction_horizon} days)...")
    engineer = FeatureEngineer(prediction_horizon=prediction_horizon)
    df_features = engineer.build_features(df, create_targets=False)

    # Get most recent record for each customer-product pair
    latest = df_features.sort_values('date').groupby(['customer_id', 'product_id']).tail(1)

    if customer_id:
        # Cast column to string to ensure matching with input string
        latest = latest[latest['customer_id'].astype(str) == str(customer_id)]

    if product_id:
        # Cast column to string to ensure matching with input string
        latest = latest[latest['product_id'].astype(str) == str(product_id)]

    return latest