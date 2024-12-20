from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import optuna
import pandas as pd
import pandas_market_calendars as mcal
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import (
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from lightweight_charts import chart, Panel, PlotDFAccessor, PlotSRAccessor
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from traceback import format_exc
from scipy.stats import entropy
import pickle
from itertools import zip_longest
import subprocess


def set_gpu_params(model):
    # Check GPU and create classifier
    has_gpu = check_gpu()
    if has_gpu:
        print("GPU is available! Using GPU acceleration.")
        gpu_params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': 0
        }
        model.set_params(**gpu_params)
    else:
        print("GPU not found. Using CPU.")
    return model

def check_gpu() -> bool:
    """Check if GPU is available via nvidia-smi"""
    try:
        subprocess.check_output('nvidia-smi')
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

#https://claude.ai/chat/dc62f18b-f293-4c7e-890d-1e591ce78763
#skew of return prediction
def create_exp_weights(num_classes):
    """
    Create exponential weights centered around middle class
    """
    middle = num_classes // 2
    weights = np.array([np.exp(i - middle) for i in range(num_classes)])
    weights = weights - np.mean(weights)  # center around 0
    return weights

def analyze_return_distribution(prob_df, actual=None):
    """
    Analyzes probability distributions from a classifier predicting return classes
    
    Parameters:
    -----------
    prob_df : pd.DataFrame
        DataFrame with probabilities for each class
        Index should be timestamps
        Columns should be class_0, class_1, etc.
    actual : pd.Series, optional
        Series with actual values, same index as prob_df
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with analysis metrics
    """
    num_classes = len(prob_df.columns)
    middle_class = num_classes // 2
    
    # Create weights once
    weights = create_exp_weights(num_classes)
    
    # Calculate metrics
    results = pd.DataFrame(index=prob_df.index)
    
    # Skew score (weighted sum of probabilities)
    results['skew_score'] = np.dot(prob_df, weights)
    
    # Uncertainty (entropy of probability distribution)
    results['uncertainty'] = prob_df.apply(entropy, axis=1)
    
    # Probability mass in different regions
    results['prob_negative'] = prob_df.iloc[:, :middle_class].sum(axis=1)
    results['prob_neutral'] = prob_df.iloc[:, middle_class]
    results['prob_positive'] = prob_df.iloc[:, (middle_class+1):].sum(axis=1)
    
    # Most probable class
    results['max_prob_class'] = prob_df.idxmax(axis=1)
    results['max_prob_value'] = prob_df.max(axis=1)
    
    if actual is not None:
        results['actual'] = actual
    
    return results

def plot_distribution_analysis(prob_df, analysis_df, actual=None, figsize=(15, 12)):
    """
    Creates comprehensive visualization of the probability distribution analysis
    
    Parameters:
    -----------
    prob_df : pd.DataFrame
        Original probability DataFrame
    analysis_df : pd.DataFrame
        Output from analyze_return_distribution
    actual : pd.Series, optional
        Actual returns
    figsize : tuple
        Figure size
    """
    fig = plt.figure(figsize=figsize)
    
    # Grid specification
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1])
    
    # 1. Skew Score Time Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(analysis_df.index, analysis_df['skew_score'], 
             label='Skew Score', color='blue', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    if actual is not None:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(actual.index, actual, 
                     label='Actual Returns', color='red', alpha=0.3)
    ax1.set_title('Return Distribution Skew Score')
    ax1.legend(loc='upper left')
    if actual is not None:
        ax1_twin.legend(loc='upper right')
    
    # 2. Probability Distribution Heatmap
    ax2 = fig.add_subplot(gs[1, :])
    sns.heatmap(prob_df.T, cmap='YlOrRd', ax=ax2)
    ax2.set_title('Probability Distribution Evolution')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Return Class')
    
    # 3. Probability Mass Distribution
    ax3 = fig.add_subplot(gs[2, 0])
    analysis_df[['prob_negative', 'prob_neutral', 'prob_positive']].plot(
        kind='area', stacked=True, ax=ax3, alpha=0.7)
    ax3.set_title('Probability Mass Distribution')
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 4. Uncertainty vs Skew Score
    ax4 = fig.add_subplot(gs[2, 1])
    scatter = ax4.scatter(analysis_df['skew_score'], 
                         analysis_df['uncertainty'],
                         c=actual if actual is not None else 'blue',
                         alpha=0.5)
    if actual is not None:
        plt.colorbar(scatter, label='Actual Returns')
    ax4.set_xlabel('Skew Score')
    ax4.set_ylabel('Uncertainty')
    ax4.set_title('Signal Strength Analysis')
    
    plt.tight_layout()
    plt.show()
    return fig

def calculate_signal_statistics(analysis_df, actual=None, 
                              skew_thresholds=(-1, 1), 
                              uncertainty_threshold=0.5):
    """
    Calculate statistics about signal reliability
    
    Parameters:
    -----------
    analysis_df : pd.DataFrame
        Output from analyze_return_distribution
    actual : pd.Series, optional
        Actual returns
    skew_thresholds : tuple
        (negative_threshold, positive_threshold)
    uncertainty_threshold : float
        Maximum uncertainty for "certain" signals
    
    Returns:
    --------
    dict
        Dictionary with signal statistics
    """
    stats = {}
    
    # Signal distribution
    stats['strong_negative'] = (analysis_df['skew_score'] < skew_thresholds[0]).mean()
    stats['strong_positive'] = (analysis_df['skew_score'] > skew_thresholds[1]).mean()
    stats['neutral'] = ((analysis_df['skew_score'] >= skew_thresholds[0]) & 
                       (analysis_df['skew_score'] <= skew_thresholds[1])).mean()
    
    # Certainty analysis
    stats['high_certainty'] = (analysis_df['uncertainty'] < uncertainty_threshold).mean()
    
    if actual is not None:
        # Calculate directional accuracy for strong signals
        strong_neg_mask = analysis_df['skew_score'] < skew_thresholds[0]
        strong_pos_mask = analysis_df['skew_score'] > skew_thresholds[1]
        
        if strong_neg_mask.any():
            stats['negative_signal_accuracy'] = (actual[strong_neg_mask] < 0).mean()
        
        if strong_pos_mask.any():
            stats['positive_signal_accuracy'] = (actual[strong_pos_mask] > 0).mean()
    
    return stats

#prediction potential + granger
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
import plotly.express as px

warnings.filterwarnings('ignore')

def analyze_and_visualize_features(file_path = None, feature_df = None):
    """
    Analyze features with comprehensive visualizations including correlation heatmaps
    """
    if file_path is not None:
        # Read data
        df = pd.read_csv(file_path, parse_dates=['Open time'])
        df.set_index('Open time', inplace=True)
    else:
        df = feature_df.copy()
    
    # Calculate correlation matrices
    pearson_corr = df.corr(method='pearson')
    spearman_corr = df.corr(method='spearman')
    
    # Figure 1: Correlation Heatmaps (2x1 grid)
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
    
    # Pearson Correlation Heatmap
    sns.heatmap(pearson_corr, 
                annot=False, 
                cmap='RdBu_r', 
                center=0,
                fmt='.2f',
                ax=ax1)
    ax1.set_title('Pearson Correlation Heatmap', fontsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Spearman Correlation Heatmap
    sns.heatmap(spearman_corr, 
                annot=False, 
                cmap='RdBu_r', 
                center=0,
                fmt='.2f',
                ax=ax2)
    ax2.set_title('Spearman Correlation Heatmap', fontsize=14)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Figure 2: Feature vs Target Correlations
    features = df.drop('target', axis=1)
    target = df['target']

    print("Features shape:", features.shape)
    print("\nFeature dtypes:")
    print(features.dtypes)

    # Convert features to numeric
    features = features.apply(pd.to_numeric, errors='coerce')
    print("\nAny NaN after conversion:")
    print(features.isna().sum())    
    
    # Calculate correlations with target
    feature_correlations = pd.DataFrame({
        'feature': features.columns,
        'pearson_corr': [pearson_corr.loc['target', col] for col in features.columns],
        'spearman_corr': [spearman_corr.loc['target', col] for col in features.columns],
        'abs_pearson': [abs(pearson_corr.loc['target', col]) for col in features.columns],
        'abs_spearman': [abs(spearman_corr.loc['target', col]) for col in features.columns]
    })
    
    # Sort by absolute Spearman correlation
    feature_correlations = feature_correlations.sort_values('abs_spearman', ascending=False)
    
    # Create visualization of top features vs target correlations
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 15))
    
    # Top 20 Features by Pearson Correlation
    sns.barplot(data=feature_correlations.head(20), 
                x='pearson_corr', 
                y='feature',
                palette='RdBu_r',
                ax=ax3)
    ax3.set_title('Top 20 Features by Pearson Correlation with Target', fontsize=12)
    ax3.set_xlabel('Pearson Correlation')
    
    # Top 20 Features by Spearman Correlation
    sns.barplot(data=feature_correlations.head(20), 
                x='spearman_corr', 
                y='feature',
                palette='RdBu_r',
                ax=ax4)
    ax4.set_title('Top 20 Features by Spearman Correlation with Target', fontsize=12)
    ax4.set_xlabel('Spearman Correlation')
    
    plt.tight_layout()
    
    # Figure 3: Top Features Scatter Plots
    fig3 = plt.figure(figsize=(20, 10))
    top_6_features = feature_correlations.head(6)['feature'].tolist()
    
    for i, feature in enumerate(top_6_features, 1):
        plt.subplot(2, 3, i)
        plt.scatter(features[feature], target, alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('Target')
        plt.title(f'Target vs {feature}\nSpearman Corr: {spearman_corr.loc["target", feature]:.3f}')

        print(f"Feature {feature} type:", type(features[feature].iloc[0]))
        features[feature] = features[feature].astype(float)
        print("Target type:", type(target.iloc[0]))
        target = target.astype(float)

        # Add trend line
        z = np.polyfit(features[feature], target, 1)
        p = np.poly1d(z)
        plt.plot(features[feature], p(features[feature]), "r--", alpha=0.8)
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\n=== Feature Analysis Summary ===")
    print(f"Total features analyzed: {len(features.columns)}")
    
    print("\nTop 10 Features by Spearman Correlation with Target:")
    summary = feature_correlations.head(10)[['feature', 'spearman_corr', 'pearson_corr']]
    print(summary.to_string(index=False))
    
    # Find features with strong correlations with target
    strong_correlations = feature_correlations[
        (feature_correlations['abs_spearman'] > 0.3) |
        (feature_correlations['abs_pearson'] > 0.3)
    ]
    
    print(f"\nFeatures with strong correlation (|correlation| > 0.3): {len(strong_correlations)}")
    
    # Identify highly correlated feature pairs among top features
    top_features = feature_correlations.head(15)['feature'].tolist()
    print("\nHighly Correlated Feature Pairs among top 15 (|correlation| > 0.8):")
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            pearson = pearson_corr.loc[top_features[i], top_features[j]]
            spearman = spearman_corr.loc[top_features[i], top_features[j]]
            if abs(pearson) > 0.8 or abs(spearman) > 0.8:
                print(f"{top_features[i]} <-> {top_features[j]}:")
                print(f"  Pearson: {pearson:.3f}")
                print(f"  Spearman: {spearman:.3f}")

    return {
        'feature_correlations': feature_correlations,
        'pearson_corr': pearson_corr,
        'spearman_corr': spearman_corr,
        'strong_correlations': strong_correlations,
        'top_features': top_features
    }

def granger_causality_test(file_path = None, features_df = None):
    """For a given dataset assumes if lagged values have predictive power.
    """
    
    if file_path is not None:
        # Read data
        df = pd.read_csv(file_path, parse_dates=['Open time'])
        df.set_index('Open time', inplace=True)
        # Assuming features_df has a datetime index and columns for each feature + 'target'
    else:
        df = features_df.copy()

    # Parameters
    max_lag = 5  # Define the maximum lag to test for causality

    # Results dictionary to store causality test results for each feature
    causality_results = {}

    # Run Granger causality tests for each feature against the target
    for feature in df.columns.drop('target'):
        try:
            # Run Granger causality test
            test_result = grangercausalitytests(
                df[['target', feature]], max_lag, verbose=False
            )
            causality_results[feature] = {
                lag: round(result[0]['ssr_ftest'][1], 4)  # Extract p-value for each lag
                for lag, result in test_result.items()
            }
        except Exception as e:
            print(f"Error testing {feature}: {e}")

    # Display results
    causality_df = pd.DataFrame(causality_results)
    print("Granger Causality Test Results (p-values):")
    print(causality_df)
    # Assuming causality_results is populated from the previous Granger causality tests
    # Convert causality_results dictionary to a DataFrame for plotting
    causality_df = pd.DataFrame(causality_results).T  # Transpose to have features as rows

    # Create a heatmap using Plotly
    fig = px.imshow(
        causality_df,
        labels=dict(x="Lag", y="Feature", color="p-value"),
        x=causality_df.columns,  # Lags
        y=causality_df.index,     # Features
        color_continuous_scale="Viridis",
        aspect="auto",
    )

    # Customize the layout
    fig.update_layout(
        title="Granger Causality Test p-values (Feature vs Target)",
        xaxis_title="Lag",
        yaxis_title="Feature",
        coloraxis_colorbar=dict(
            title="p-value",
            tickvals=[0.01, 0.05, 0.1],
            ticktext=["0.01", "0.05", "0.1"]
        )
    )

    fig.show()

class Bars:
    """
    Class to represent a number of bars for walk forward window calculation.
    
    Parameters:
        count (int): Number of bars
    """
    def __init__(self, count: int):
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Bars count must be a positive integer")
        self.count = count

    def __repr__(self):
        return f"Bars({self.count})"

@dataclass
class ModelConfig:
    """Configuration for the trading model"""
    train_period: int | pd.Timedelta | Bars = 10
    test_period: int | pd.Timedelta | Bars = 1
    reoptimize_frequency: Optional[int] = 5 #hypertuning every Nth iteration 
    test_on_train: bool = False
    forward_bars: int = 5
    target_threshold: float = 1.005 # upper pct threshold for target (1.005 = up by 0.5%)
    target_direction: str = "rise"
    target_returns_ewm: int = 10
    target_reversal_threshold: float = 0.3 #How much retracement to allow 0.3=30%retracement terminates the window
    target_min_bars: int = 2 ## minimum bars to consider a valid movement
    target_min_profit_threshold: float = 0.0015  # 0.15% minimum profit threshold to maintain
    target_max_drawdown: float = 0.002 # 0.002 = 0.2% maximum drawdown to allow

    volatility_window: int = 100
    model_type: str = 'classifier'
    n_classes: int = 3
    ma_lengths: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    warm_up_period: int = None # Number of bars to attach before test period
    features_fib_max_lookback: pd.Timedelta = pd.Timedelta(hours=1) # maximum features lookback
    features_fib_max_windows: int = None # limit features windows sizes to certain count
    optuna_trials:int = 3
    summary_analysis_profit_th: float = 1
    summary_per_iteration: bool = True
    summary_slippage_pct: float = 0.1
    importance_per_iteration: bool = True

class BaseFeatureBuilder(ABC):
    """Abstract base class for feature engineering"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.generated_features = set()
        
    @abstractmethod
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features from input data"""
        pass
    
    def get_feature_descriptions(self) -> dict:
        """Return descriptions of features"""
        return self._feature_descriptions
    
    @abstractmethod
    def create_target(self, df: pd.DataFrame, train_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """Creates target variables"""
        pass

    def remove_crossday_targets(self, target: pd.Series, df: pd.DataFrame, future_bars: int, replace_value = None) -> pd.Series:
        """
        Remove targets that cross day boundaries for intraday trading.
        
        Parameters:
        -----------
        target : pd.Series
            Original target series with log returns
        df : pd.DataFrame
            Original dataframe with datetime index
        future_bars : int
            Number of forward bars used for target calculation
        replace_value : float, optional
            Value to replace cross-day targets with (for example class 4 means zero return)
            
        Returns:
        --------
        pd.Series
            Target series with cross-day targets set to NaN
        """        
        # Get dates from index
        dates = df.index.date
        
        # Create mask for same-day targets
        future_dates = df.index.date[future_bars:]
        current_dates = dates[:-future_bars]
        same_day_mask = (future_dates == current_dates)
        
        # Pad the mask to match original length
        full_mask = np.pad(same_day_mask, (0, future_bars), constant_values=False)
        
        # Apply mask to keep only intraday targets
        target_cleaned = target.copy()
        target_cleaned[~full_mask] = np.nan
        
        if replace_value is not None:
            target_cleaned[~full_mask] = replace_value
            #print number of replaced values
            print(f"Number of replaced values: {len(target_cleaned[~full_mask])}") 

        # Calculate percentage of valid targets
        valid_targets_pct = (target_cleaned.notna().sum() / len(target_cleaned)) * 100
        print(f"Percentage of valid intraday targets: {valid_targets_pct:.2f}%")

        return target_cleaned

class LibraryTradingModel:
    """Main trading model implementation with configuration-based setup"""
    
    def __init__(self, config: Optional[ModelConfig] = None, feature_builder: Optional[BaseFeatureBuilder] = None, load_model: str = None, save_model = False):
        self.config = config or ModelConfig()
        self.feature_builder = feature_builder
        self.scaler = StandardScaler()
        self.best_params = None
        self.study = None
        self.def_params = {'n_estimators': 192, 'max_depth': 4, 'learning_rate': 0.11955492653616603, 'min_child_weight': 6, 'subsample': 0.7593587243666793, 'colsample_bytree': 0.841538282739158, 'gamma': 9.747926761292942e-06, 'lambda': 2.389116689874295, 'alpha': 0.4036592961514103}
        self.use_model = None
        if load_model is not None:
            self.use_model, self.scaler = self.load_pickled(load_model)
        self.save_model = save_model

    def load_pickled(self, file):
        #LOAD MODEL/SCALER COMBO
        # Load both scaler and model
        with open(file, "rb") as f:
            data = pickle.load(f)

        print("Model LOADED from " + file)

        return data["model"], data["scaler"]

    def save_pickled(self, iteration, model):
        file = "model_scaler_" + str(iteration) + ".pkl"

        # Save both scaler and model
        with open(file, "wb") as f:
            pickle.dump({"model": model, "scaler": self.scaler}, f)
        print("Model SAVED as " + file)
    def get_date_windows(self, data: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Calculate date windows for training and testing using market days.
        Uses NYSE calendar for market days calculation.
        Handles timezone-aware input data (US/Eastern).
        """
        import pandas_market_calendars as mcal
        
        # Get NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        
        # Get all valid market days in our data range
        schedule = nyse.schedule(
            start_date=data.index[0].tz_convert('America/New_York'),
            end_date=data.index[-1].tz_convert('America/New_York')
        )
        
        # Convert schedule to US/Eastern to match input data
        market_days = pd.DatetimeIndex(schedule.index).tz_localize('US/Eastern')
        
        windows = []
        start_idx = market_days.searchsorted(data.index[0]) #iterate from second day, so we can attach warmup
        end_idx = market_days.searchsorted(data.index[-1])
        current_idx = start_idx
        
        while True:
            # Calculate indices for train and test windows
            train_end_idx = current_idx + self.config.train_period
            test_end_idx = train_end_idx + self.config.test_period
            
            # Break if we've reached the end of data
            if test_end_idx >= end_idx:
                break
            
            # Get the actual dates from market days
            current_start = market_days[current_idx]
            train_end = market_days[train_end_idx]
            test_end = market_days[test_end_idx]
            
            windows.append((current_start, train_end, train_end, test_end))
            
            # Move forward by test period in market days
            current_idx += self.config.test_period
            
        return windows

    def get_time_windows(self, data: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Calculate time windows for training and testing using the data's index.
        Supports three types of periods:
        - Integer market days
        - Time periods (pandas.Timedelta)
        - Number of bars (Bars class)
        """
        from pandas import Timedelta
        
        def get_bars_offset(current_idx: int, bars: Bars) -> Optional[pd.Timestamp]:
            if current_idx + bars.count >= len(data.index):
                return None
            return data.index[current_idx + bars.count]
        
        def get_timedelta_offset(start_date: pd.Timestamp, offset: Timedelta) -> Optional[pd.Timestamp]:
            """Handle timedelta offsets using direct datetime operations"""
            target_time = start_date + offset
            next_idx = data.index.searchsorted(target_time)
            
            # If we're beyond data range
            if next_idx >= len(data.index):
                return None
                
            # Get the actual available timestamp
            actual_time = data.index[next_idx]
            
            # Check if the found timestamp is within acceptable range
            # (not more than one original frequency away from target)
            # if actual_time - target_time > offset * 0.1:  # 10% tolerance
            #     return None
                
            return actual_time
        
        def get_market_day_offset(start_date: pd.Timestamp, offset_days: int) -> Optional[pd.Timestamp]:
            start_idx = market_days.searchsorted(start_date)
            end_idx = start_idx + offset_days
            
            if end_idx >= len(market_days):
                return None
                
            target_date = market_days[end_idx]
            # Check if we have enough data points after the target date
            target_idx = data.index.searchsorted(target_date)
            if target_idx >= len(data.index):
                return None
                
            return target_date
        
        # Rest of the function remains the same...
        if isinstance(self.config.train_period, Bars):
            def get_offset(start_date, period):
                idx = data.index.searchsorted(start_date)
                return get_bars_offset(idx, period)
            training_period = self.config.train_period
            testing_period = self.config.test_period
            
        elif isinstance(self.config.train_period, Timedelta):
            get_offset = get_timedelta_offset
            training_period = self.config.train_period
            testing_period = self.config.test_period
            
        elif isinstance(self.config.train_period, (int, float)):
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar('NYSE')
            
            schedule = nyse.schedule(
                start_date=data.index[0].tz_convert('America/New_York'),
                end_date=data.index[-1].tz_convert('America/New_York')
            )
            market_days = pd.DatetimeIndex(schedule.index).tz_localize('US/Eastern')
            
            get_offset = get_market_day_offset
            training_period = self.config.train_period
            testing_period = self.config.test_period
        
        else:
            raise ValueError(
                "train_period and test_period must be either:\n"
                "- Integer (market days)\n"
                "- pandas.Timedelta (time period)\n"
                "- Bars (number of bars)"
            )
        
        windows = []
        if isinstance(self.config.train_period, (int, float)):
            current_start = market_days[0]
        else:
            current_start = data.index[0]
        
        while True:
            train_end = get_offset(current_start, training_period)
            if train_end is None:
                break
                
            test_end = get_offset(train_end, testing_period)
            if test_end is None:
                break
            
            windows.append((current_start, train_end, train_end, test_end))
            
            next_start = get_offset(current_start, testing_period)
            if next_start is None:
                break
            current_start = next_start
        
        return windows

    def create_model(self, trial=None):
        """Create XGBoost model with either default or Optuna-suggested parameters"""
        if self.config.model_type == 'classifier':
            num_class = self.config.n_classes if self.config.n_classes > 2 else None #for binary no num_class
            from xgboost import XGBClassifier
            if trial is None:
                return XGBClassifier(n_estimators=100, random_state=42, num_class=num_class)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
                    "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
                    'random_state': 42
                }
                return XGBClassifier(**params, num_class=num_class)
        else:
            from xgboost import XGBRegressor
            if trial is None:
                return XGBRegressor(n_estimators=100, random_state=42)
            else:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'random_state': 42
                }
                return XGBRegressor(**params)

    def run_rolling_window(self, data: pd.DataFrame, num_iterations: Optional[int] = None) -> Dict:
        """Run the model using a rolling window approach"""
        windows = self.get_time_windows(data)
        if num_iterations:
            windows = windows[:num_iterations]
        
        all_results = {}

        # Reset best_params for each rolling window run
        self.best_params = None

        # Add hyperparameter reoptimization frequency
        reoptimize_every_n_iterations = self.config.reoptimize_frequency or 5

        #number of warm up bars for each iteration
        warm_period = self.config.warm_up_period if self.config.warm_up_period is not None else 0
        print("Warmup period:", warm_period)

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # If warm_period is 0, use original timestamps, otherwise add warm-up period
            if warm_period > 0:
                train_warmup_data = data[data.index < train_start].tail(warm_period)
                train_start_with_warmup = train_warmup_data.index[0] if not train_warmup_data.empty else train_start
                
                test_warmup_data = data[data.index < test_start].tail(warm_period)
                test_start_with_warmup = test_warmup_data.index[0] if not test_warmup_data.empty else test_start
            else:
                train_start_with_warmup = train_start
                test_start_with_warmup = test_start
            
            train_mask = (data.index >= train_start_with_warmup) & (data.index < train_end)
            test_mask = (data.index >= test_start_with_warmup) & (data.index < test_end)
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            min_required_bars = max(20, self.config.forward_bars + 1)
            if len(train_data) < min_required_bars or len(test_data) < 1:
                print(f"Skipping iteration {i}: Insufficient data")
                continue

            # Reoptimize hyperparameters periodically
            if i % reoptimize_every_n_iterations == 0:
                self.best_params = None  # Force reoptimization

            results, model = self.run_iteration(train_data, test_data, i)
            
            if results is not None:
                if self.save_model: #save model if required
                    self.save_pickled(i, model)

                all_results[i] = {
                    'train_period': (train_start, train_end),
                    'test_period': (test_start, test_end),
                    'results': results,
                    'model': model,
                    'hyperparameters': self.best_params.copy() if self.best_params else None
                }
        
        return all_results

    def optimize_hyperparameters(self, X_train, y_train, n_trials=2):
        """Run Optuna hyperparameter optimization using time series cross-validation"""
        import optuna
        from sklearn.model_selection import TimeSeriesSplit
        
        print("\nStarting hyperparameter optimization...")
        
        # Calculate appropriate number of splits based on data size
        total_samples = len(X_train)
        test_size = int(len(X_train) * 0.2)  # 20% validation size
        gap = self.config.forward_bars
        
        # Calculate maximum possible splits
        available_samples = total_samples
        max_splits = 0
        while available_samples >= 2 * test_size + gap:  # Need at least 2x test_size (for train and test) plus gap
            available_samples -= (test_size + gap)
            max_splits += 1
        
        n_splits = min(3, max_splits)  # Use at most 3 splits, or fewer if data doesn't allow more
        
        print(f"Using {n_splits} splits for time series cross-validation")
        print(f"Total samples: {total_samples}, Test size: {test_size}, Gap: {gap}")
        
        # Create time series cross-validation splits
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap
        )
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                "lambda": trial.suggest_float("lambda", 1e-3, 10, log=True),
                "alpha": trial.suggest_float("alpha", 1e-3, 10, log=True),
                'random_state': 42
            }
            
            # Store scores from each fold
            fold_scores = []
            
            # Time series cross-validation
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # Create and train model
                if self.config.model_type == 'classifier':
                    model = XGBClassifier(**params)
                    if self.config.n_classes > 2:
                        model.set_params(num_class=self.config.n_classes)
                else:
                    model = XGBRegressor(**params)

                model = set_gpu_params(model)

                # Determine class weights dynamically
                class_counts = np.bincount(y_fold_train)  # Counts the occurrences of each class
                class_weights = {i: sum(class_counts) / (len(class_counts) * count) for i, count in enumerate(class_counts)}

                # Check number of classes
                if self.config.n_classes == 2:
                    # For binary classification, use scale_pos_weight
                    n_0 = class_counts[0]
                    n_1 = class_counts[1]
                    scale_pos_weight = n_0 / n_1
                    model.set_params(scale_pos_weight=scale_pos_weight)
                else:
                    # For multiclass classification, use sample_weight
                    sample_weights = [class_weights[label] for label in y_fold_train]

                print("Model Training...")
                if self.config.n_classes == 2:
                    # Train with early stopping
                    model.fit(
                        X_fold_train, 
                        y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        #early_stopping_rounds=50,
                        #verbose=2
                    )
                else:
                    # Train with early stopping
                    model.fit(
                        X_fold_train, 
                        y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        sample_weight=sample_weights
                        #early_stopping_rounds=50,
                        #verbose=2
                    )

                # Calculate score
                if self.config.model_type == 'classifier':
                    pred = model.predict(X_fold_val)
                    score = accuracy_score(y_fold_val, pred)
                else:
                    pred = model.predict(X_fold_val)
                    score = -mean_squared_error(y_fold_val, pred, squared=False)
                
                fold_scores.append(score)
            
            # Return mean score across folds
            return np.mean(fold_scores)
        
        print("init optuna study")

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=20,
                interval_steps=10
            )
        )
        
        print("starting optimization")

        # Run optimization
        study.optimize(objective, n_trials=n_trials)
        
        self.study = study
        self.best_params = study.best_params
        
        print("\nHyperparameter Optimization Results:")
        print(f"Best score: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for param, value in study.best_params.items():
            print(f"{param}: {value}")
            
        return study.best_params

    def run_iteration(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                     iteration_num: int) -> Tuple[Optional[pd.DataFrame], Optional[object]]:
        """Run a single iteration of training and testing with optional hyperparameter optimization"""
        try:
            train_features = None
            train_target = None
            train_cols = []
            test_cols = []
            print(f"\nProcessing iteration {iteration_num}")
            print(f"Training: {train_data.index[0]} to {train_data.index[-1]} : {train_data.shape}")
            print(f"Testing: {test_data.index[0]} to {test_data.index[-1]} : {test_data.shape}")
            
            # Create features on combined and then split it again
            print("feature generating started.")
            train_features, feature_cols = self.feature_builder.prepare_features(train_data)
            print("Features created. Target starting")
            train_target = self.feature_builder.create_target(train_features)
            print("Target created")

            if self.use_model is None: #when model is provided, dont prepare train data, only test
                X_train = train_features
                y_train = train_target
                
                print("TRAIN-----")
                print(f"X_train shape: {X_train.shape}", X_train.index[[0,-1]])
                print(f"y_train shape: {y_train.shape}", y_train.index[[0,-1]])
                print("Removing NaNs")
                
                # Remove NaN values or infinite values
                y_train = y_train.replace([np.inf, -np.inf], np.nan)
                mask_train = ~y_train.isna()
                X_train = X_train[mask_train]
                y_train = y_train[mask_train]
                
                print(f"X_train shape after cleaning: {X_train.shape}", X_train.index[[0,-1]])
                print(f"y_train shape after cleaning: {y_train.shape}", y_train.index[[0,-1]])
                print(f"X_train columns: {X_train.columns}")
                train_cols = set(X_train.columns)
                train_columns = X_train.columns
                
                if len(X_train) < self.config.forward_bars + 1:
                    print(f"Warning: Iteration {iteration_num} - Insufficient training data")
                    return None, None
                            
                # Scale features
                print("Scaling features...")
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            
                # Run hyperparameter optimization if not done yet
                if self.best_params is None and self.config.optuna_trials is not None and self.config.optuna_trials > 0:
                    print("optimization started...")
                    self.optimize_hyperparameters(X_train_scaled, y_train, self.config.optuna_trials)
                
                # Create and train model with best parameters
                model = self.create_model()
                if self.best_params:
                    model.set_params(**self.best_params)
                else:
                    print("Using default hyperparameters",self.def_params)
                    model.set_params(**self.def_params)
                    
                model = set_gpu_params(model)

                model.set_params(verbosity=1)

                # Determine class weights dynamically
                class_counts = np.bincount(y_train)  # Counts the occurrences of each class
                class_weights = {i: sum(class_counts) / (len(class_counts) * count) for i, count in enumerate(class_counts)}

                # Check number of classes
                if self.config.n_classes == 2:
                    # For binary classification, use scale_pos_weight
                    n_0 = class_counts[0]
                    n_1 = class_counts[1]
                    scale_pos_weight = n_0 / n_1
                    model.set_params(scale_pos_weight=scale_pos_weight)
                else:
                    # For multiclass classification, use sample_weight
                    sample_weights = [class_weights[label] for label in y_train]

                print("Model Training...")
                if self.config.n_classes == 2:
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                print("Using PROVIDED MODEL and SCALER")
                model = self.use_model

                model = set_gpu_params(model)
            
            print("TEST-----")
            if self.config.test_on_train:
                print("TESTED ON TRAIN DATA!")
                X_test = train_features
                y_test = train_target
            else:
                test_features, features_cols = self.feature_builder.prepare_features(test_data)
                X_test = test_features
                y_test = self.feature_builder.create_target(test_features, train_data=train_features)
         
            print(f"X_test shape: {X_test.shape}", X_test.index[[0,-1]])
            print(f"y_test shape: {y_test.shape}", y_test.index[[0,-1]])
            print("Removing NaNs")
            
            # Remove NaN values or infinite values
            y_test = y_test.replace([np.inf, -np.inf], np.nan)
            mask_test = ~y_test.isna()
            X_test = X_test[mask_test]
            y_test = y_test[mask_test]
            
            print(f"X_test shape after cleaning: {X_test.shape}", X_test.index[[0,-1]])
            print(f"y_test shape after cleaning: {y_test.shape}", y_test.index[[0,-1]])
            print("X_test columns:", X_test.columns)
            test_cols = set(X_test.columns)

            #Trimming the warmup period if needed
            warm_period = self.config.warm_up_period if self.config.warm_up_period is not None else 0
            if warm_period > 0:
                print(f"Trimming warmup period... {warm_period}")
                X_test = X_test.iloc[warm_period:]
                y_test = y_test.iloc[warm_period:]
                print(f"X_test shape after trimming: {X_test.shape}", X_test.index[[0,-1]])
                print(f"y_test shape after trimming: {y_test.shape}", y_test.index[[0,-1]])

            if self.use_model is None:
                # Find columns in test but not in train
                extra_in_test = test_cols - train_cols
                print("Extra columns in X_test:", extra_in_test)

                # Find columns in train but not in test 
                extra_in_train = train_cols - test_cols
                print("Extra columns in X_train:", extra_in_train)

                # Reorder X_test columns to match, excpet when model was provided
                X_test = X_test[train_columns]
            else:
                # Assuming X_test is a DataFrame, ensure ordering
                X_test = X_test[self.scaler.feature_names_in_]


            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            # Make predictions
            predictions = model.predict(X_test_scaled)

            if self.config.model_type == 'classifier':
                predictions_proba = model.predict_proba(X_test_scaled)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'predicted': predictions,
                'actual': y_test
            }, index=X_test.index)

            if "close" in X_test.columns:
                results["close"] = X_test["close"]

            if self.config.model_type == 'regressor':
                self.iteration_summary_regressor(results, model, iteration_num)
            else:
                self.iteration_summary_classifier(results, model, predictions_proba, iteration_num)

            if self.config.importance_per_iteration:
                self.plot_feature_importance(model)

            return results, model
            
        except Exception as e:
            raise e
            print(f"Error in iteration {iteration_num}: {str(e)} - {format_exc()}")
            return None, None


    def calculate_probability_skew(self, predict_proba_output, weights=None):
        """
        Calculate probability skew metrics for each individual prediction.
        """
        num_class = self.config.n_classes
        # Validate input dimensions
        probs = np.array(predict_proba_output)
        if probs.shape[1] != num_class:
            raise ValueError(f"predict_proba_output shape {probs.shape} doesn't match num_class {num_class}")
        
        # Create linear weights if not provided
        if weights is None:
            weights = np.linspace(0, num_class - 1, num_class)
        elif len(weights) != num_class:
            raise ValueError(f"weights length {len(weights)} doesn't match num_class {num_class}")
        
        # Calculate weighted score for each prediction
        weighted_probs = np.sum(probs * weights, axis=1)
        
        # Calculate basic metrics for each prediction
        max_class_prob = np.max(probs, axis=1)
        positive_class_prob = probs[:, -1]
        predicted_class = np.argmax(probs, axis=1)
        
        # Calculate skewness
        center = (num_class - 1) / 2
        distances = np.arange(num_class) - center
        skewness = np.sum(probs * distances, axis=1) / center
        
        # Calculate high/low ratio
        mid_point = num_class // 2
        higher_probs = np.sum(probs[:, mid_point:], axis=1)
        lower_probs = np.sum(probs[:, :mid_point], axis=1)
        high_low_ratio = np.where(lower_probs > 0, higher_probs / lower_probs, higher_probs)
        
        # Calculate Progressive Growth Score (excluding class 0)
        def calculate_progress_score(prob_row):
            # Extract probabilities excluding class 0
            probs_excl_0 = prob_row[1:]
            n = len(probs_excl_0)
            
            # Compare each probability with all previous ones
            total_comparisons = 0
            positive_growths = 0
            
            for i in range(1, n):
                # Compare with all previous probabilities
                comparisons = probs_excl_0[i] > probs_excl_0[:i]
                positive_growths += np.sum(comparisons)
                total_comparisons += i
            
            # Normalize score between -1 and 1
            # -1: perfect reverse stairs, 0: random, 1: perfect stairs
            return (2 * positive_growths / total_comparisons - 1) if total_comparisons > 0 else 0
        
        progress_scores = np.array([calculate_progress_score(p) for p in probs])
        
        # Combine metrics into a single array
        metrics = np.column_stack([
            weighted_probs,                     # Weighted probability score
            max_class_prob,                     # Maximum probability
            positive_class_prob,                # Probability of highest class
            predicted_class,                    # Predicted class
            (predicted_class == (num_class-1)), # Binary indicator for highest class
            skewness,                          # Skewness towards higher classes
            high_low_ratio,                    # Ratio of higher to lower class probabilities
            progress_scores                    # Progressive Growth Score
        ])
        
        # Create column names for reference
        metric_names = ['weighted_score', 'max_prob', 'positive_prob', 
                    'predicted_class', 'is_positive', 'skewness', 
                    'high_low_ratio', 'progress_score']
        
        return metrics, metric_names

    def iteration_summary_classifier(self,results, model, predictions_proba, iteration_num):
        """
        Analyze classifier results with focus on directional confidence and class probabilities
        
        Parameters:
        - results: DataFrame with 'predicted' and 'actual' columns
        - model: trained XGBoost classifier
        - predictions_proba: probability predictions for each class
        - iteration_num: current iteration number
        """


        def evaluate_directional_accuracy(results, predictions_proba, confidence_thresholds={'high': 0.7, 'medium': 0.5}):
            """
            Evaluate directional prediction accuracy with confidence thresholds.
            
            Parameters:
            - results: DataFrame with 'predicted' and 'actual' columns
            - predictions_proba: probability predictions for each class
            - confidence_thresholds: dict with threshold levels
            
            Returns:
            - Dictionary containing evaluation metrics and data for visualization
            """
            import numpy as np
            import pandas as pd
            
            # Get number of classes
            n_classes = predictions_proba.shape[1]
            
            # Create DataFrame with probabilities
            class_names = [f'class_{i}' for i in range(n_classes)]
            prob_df = pd.DataFrame(predictions_proba, columns=class_names)
            prob_df.index = results.index
            
            # Extract extreme class probabilities
            neg_probs = prob_df['class_0']  # highest negative returns
            pos_probs = prob_df[f'class_{n_classes-1}']  # highest positive returns
            
            # Determine directional predictions based on confidence thresholds
            directional_preds = pd.Series(index=results.index, dtype='str')
            directional_preds[neg_probs >= confidence_thresholds['high']] = 'strong_negative'
            directional_preds[pos_probs >= confidence_thresholds['high']] = 'strong_positive'
            directional_preds[neg_probs.between(confidence_thresholds['medium'], confidence_thresholds['high'])] = 'weak_negative'
            directional_preds[pos_probs.between(confidence_thresholds['medium'], confidence_thresholds['high'])] = 'weak_positive'
            directional_preds[directional_preds.isnull()] = 'neutral'
            
            # Determine actual directions
            actual_dirs = pd.Series(index=results.index, dtype='str')
            actual_dirs[results['actual'] == 0] = 'negative'
            actual_dirs[results['actual'] == n_classes-1] = 'positive'
            actual_dirs[actual_dirs.isnull()] = 'neutral'
            
            # Calculate penalties
            penalties = pd.Series(index=results.index, dtype='float')
            
            # Define penalty weights
            penalty_matrix = {
                ('strong_negative', 'positive'): -2.0,
                ('strong_positive', 'negative'): -2.0,
                ('weak_negative', 'positive'): -1.5,
                ('weak_positive', 'negative'): -1.5,
                ('neutral', 'positive'): -0.5,
                ('neutral', 'negative'): -0.5,
                ('strong_negative', 'negative'): 1.0,
                ('strong_positive', 'positive'): 1.0,
                ('weak_negative', 'negative'): 0.5,
                ('weak_positive', 'positive'): 0.5,
            }
            
            # Apply penalties
            for pred_dir, actual_dir in penalty_matrix.keys():
                mask = (directional_preds == pred_dir) & (actual_dirs == actual_dir)
                penalties[mask] = penalty_matrix[(pred_dir, actual_dir)]
            
            # Fill remaining combinations with neutral penalty (0)
            penalties.fillna(0, inplace=True)
            
            # Calculate metrics
            metrics = {
                'total_score': penalties.sum(),
                'avg_score': penalties.mean(),
                'high_conf_accuracy': (
                    ((directional_preds == 'strong_negative') & (actual_dirs == 'negative')) |
                    ((directional_preds == 'strong_positive') & (actual_dirs == 'positive'))
                ).mean(),
                'med_conf_accuracy': (
                    ((directional_preds == 'weak_negative') & (actual_dirs == 'negative')) |
                    ((directional_preds == 'weak_positive') & (actual_dirs == 'positive'))
                ).mean(),
                'confusion_data': pd.crosstab(directional_preds, actual_dirs),
                'confidence_data': {
                    'correct_neg': neg_probs[(directional_preds == 'strong_negative') & (actual_dirs == 'negative')],
                    'incorrect_neg': neg_probs[(directional_preds == 'strong_negative') & (actual_dirs != 'negative')],
                    'correct_pos': pos_probs[(directional_preds == 'strong_positive') & (actual_dirs == 'positive')],
                    'incorrect_pos': pos_probs[(directional_preds == 'strong_positive') & (actual_dirs != 'positive')]
                }
            }
            
            return metrics

        def plot_directional_analysis(metrics, iteration_num):
            """
            Create visualization plots for directional analysis results
            
            Parameters:
            - metrics: Dictionary containing evaluation metrics from evaluate_directional_accuracy
            - iteration_num: Current iteration number
            """
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create a figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # 1. Confusion Matrix Heatmap
            plt.subplot(2, 2, 1)
            sns.heatmap(metrics['confusion_data'], 
                        annot=True, 
                        fmt='d',
                        cmap='YlOrRd')
            plt.title(f'Directional Prediction Confusion Matrix\nIteration {iteration_num}')
            
            # 2. Confidence Distribution Plot
            plt.subplot(2, 2, 2)
            
            # Plot correct predictions
            if len(metrics['confidence_data']['correct_neg']) > 0:
                sns.kdeplot(data=metrics['confidence_data']['correct_neg'], 
                        label='Correct Negative', color='blue', alpha=0.6)
            if len(metrics['confidence_data']['correct_pos']) > 0:
                sns.kdeplot(data=metrics['confidence_data']['correct_pos'], 
                        label='Correct Positive', color='green', alpha=0.6)
            
            # Plot incorrect predictions
            if len(metrics['confidence_data']['incorrect_neg']) > 0:
                sns.kdeplot(data=metrics['confidence_data']['incorrect_neg'], 
                        label='Incorrect Negative', color='red', alpha=0.6)
            if len(metrics['confidence_data']['incorrect_pos']) > 0:
                sns.kdeplot(data=metrics['confidence_data']['incorrect_pos'], 
                        label='Incorrect Positive', color='orange', alpha=0.6)
            
            plt.title('Confidence Distribution by Prediction Outcome')
            plt.xlabel('Prediction Confidence')
            plt.ylabel('Density')
            plt.legend()
            
            # 3. Accuracy Metrics Bar Plot
            plt.subplot(2, 2, 3)
            metrics_to_plot = {
                'High Conf\nAccuracy': metrics['high_conf_accuracy'],
                'Med Conf\nAccuracy': metrics['med_conf_accuracy'],
                'Overall\nScore': metrics['avg_score']
            }
            
            plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
            plt.title('Performance Metrics')
            plt.ylabel('Score')
            
            # Add text annotations
            plt.text(0.1, 0.95, f'Total Score: {metrics["total_score"]:.2f}', 
                    transform=plt.gca().transAxes)
            
            plt.show()
            return

        class_names = [f'class_{c}' for c in model.classes_]
        #class_names = [f'class_{i}' for i in range(len(predictions_proba[0]))]
        print(class_names)
        prob_df = pd.DataFrame(predictions_proba, columns=class_names)
        prob_df.index = results.index

        #calculate positivity skew (probability skew towards positive class 0 to N, where N is most positive)
        pos_skew_metrics, names = self.calculate_probability_skew(predictions_proba)
        pos_skew_df = pd.DataFrame(pos_skew_metrics, columns=names, index=results.index)

        # # Verification step
        # def verify_xgb_predictions(model, predictions, predictions_proba):
        #     # Get predicted class from probabilities
        #     pred_from_proba = model.classes_[np.argmax(predictions_proba, axis=1)]
            
        #     # Check if they match
        #     matches = (predictions == pred_from_proba)
        #     if not np.all(matches):
        #         print("Warning: Predictions don't match probability argmax")
        #         print("Mismatched indices:", np.where(~matches)[0])
        #         print("\nSample of mismatches:")
        #         mismatch_idx = np.where(~matches)[0][:5]  # Show first 5 mismatches
        #         for idx in mismatch_idx:
        #             print(f"\nIndex {idx}:")
        #             print(f"Prediction: {predictions[idx]}")
        #             print(f"Prediction from proba: {pred_from_proba[idx]}")
        #             print("Probabilities:")
        #             for c, p in zip(model.classes_, predictions_proba[idx]):
        #                 print(f"class_{c}: {p:.3f}")
        #     else:
        #         print("✓ Predictions match probability argmax")
            
        #     return pred_from_proba

        # # Run verification
        # pred_from_proba = verify_xgb_predictions(model, results["predicted"], predictions_proba)

        # # You can add the verification to results if needed
        # results['pred_from_proba'] = pred_from_proba

        # # Print sample to verify
        # print("\nResults sample:")
        # print(results.head())
        # print("\nProbabilities sample:")
        # print(prob_df.head())


        # Add directional analysis
        # dir_metrics = evaluate_directional_accuracy(
        #     results, 
        #     predictions_proba,
        #     confidence_thresholds={'high': 0.6, 'medium': 0.3}
        # )

        # print(dir_metrics)

        # plot_directional_analysis(
        #     metrics=dir_metrics,
        #     iteration_num=iteration_num)
        
        analysis_df = analyze_return_distribution(prob_df, results["actual"])
        fig = plot_distribution_analysis(prob_df, analysis_df, results["actual"])
        stats = calculate_signal_statistics(analysis_df, results["actual"])

        # Print statistics
        print("\nSignal Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2%}")

        if self.config.summary_per_iteration:
            Panel(
                #auto_scale=[prob_df],
                histogram=[],
                right=[(results["close"], "close") if "close" in results.columns else ()],
                left=[],
                middle1=[(results["predicted"],"predicted"),(results["actual"],"actual"),(prob_df,)],
                middle2=[(pos_skew_df,)]
            ).chart(size="s", precision=6, title=f"Iteration {iteration_num} classes:{self.config.n_classes} forward_bars:{self.config.forward_bars}")   

        num_classes = self.config.n_classes
        
        # Add probability columns to results
        for i in range(num_classes):
            results[f'prob_class_{i}'] = predictions_proba[:, i]
        
        # Calculate class groupings
        if num_classes < 2:
            print("Must have at least 2 classes")
            raise ValueError("Must have at least 2 classes")
        
        # For 2 classes: negative=[0], positive=[1], no neutral
        # For 3 classes: negative=[0], neutral=[1], positive=[2]
        # For 4 classes: negative=[0,1], positive=[2,3], no neutral
        # For 5 classes: negative=[0,1], neutral=[2], positive=[3,4]
        # And so on...
        
        negative_classes = list(range(num_classes // 2 if num_classes > 2 else 1))
        positive_classes = list(range(num_classes - (num_classes // 2 if num_classes > 2 else 1), num_classes))
        
        # Calculate if there should be a neutral class
        has_neutral = num_classes > 2 and num_classes % 2 == 1
        neutral_class = [num_classes // 2] if has_neutral else []
        
        # Calculate directional probabilities
        results['prob_negative'] = results[[f'prob_class_{i}' for i in negative_classes]].sum(axis=1)
        if has_neutral:
            results['prob_neutral'] = results[f'prob_class_{neutral_class[0]}']
        else:
            results['prob_neutral'] = 0.0
        results['prob_positive'] = results[[f'prob_class_{i}' for i in positive_classes]].sum(axis=1)
        
        # Define direction mapping function
        def get_direction(x):
            if x in negative_classes:
                return 'negative'
            elif x in positive_classes:
                return 'positive'
            return 'neutral'

        # Calculate directional predictions
        results['predicted_direction'] = results['predicted'].map(get_direction)
        results['actual_direction'] = results['actual'].map(get_direction)
        results['direction_correct'] = results['predicted_direction'] == results['actual_direction']
    
        # 1. Print Summary Statistics
        print(f"\n=== Iteration {iteration_num} Summary ===")
        print("\nClass Distribution:")
        print("Actual class distribution:")
        print(results['actual'].value_counts().sort_index())
        print("\nPredicted class distribution:")
        print(results['predicted'].value_counts().sort_index())
        
        print("\nDirectional Distribution:")
        print("Actual direction distribution:")
        print(results['actual_direction'].value_counts())
        print("\nPredicted direction distribution:")
        print(results['predicted_direction'].value_counts())
        
        print("\nAccuracy Metrics:")
        print("Overall Accuracy:", (results['predicted'] == results['actual']).mean())
        print("Directional Accuracy:", results['direction_correct'].mean())
        
        print("New confusion matrix")


        def plot_threshold_confusion_matrices(results, predictions_proba, thresholds=[0.3, 0.5, 0.8], num_classes=8):
            """
            Plot confusion matrices for different probability thresholds
            """
            # Calculate subplot dimensions
            plots_per_row = 3
            num_rows = (len(thresholds) + plots_per_row - 1) // plots_per_row  # Ceiling division
            num_cols = min(len(thresholds), plots_per_row)
            
            # Create figure with extra spacing between subplots
            fig = plt.figure(figsize=(7*num_cols, 6*num_rows))
            gs = fig.add_gridspec(num_rows, num_cols, hspace=0.4, wspace=0.3)
            
            for idx, threshold in enumerate(thresholds):
                # Calculate subplot position
                row = idx // plots_per_row
                col = idx % plots_per_row
                ax = fig.add_subplot(gs[row, col])
                
                # Create masked predictions where low confidence predictions are marked as -1
                predicted_classes = np.full(len(predictions_proba), -1)
                max_probs = np.max(predictions_proba, axis=1)
                confident_mask = max_probs >= threshold
                
                # Debug - print first few samples
                # print(f"\nThreshold: {threshold}")
                # print("First 5 probabilities arrays:")
                # for i in range(5):
                #     if confident_mask[i]:
                #         prob_array = predictions_proba[i]
                #         pred_idx = np.argmax(prob_array)
                #         print(f"\nSample {i}:")
                #         print(f"Probabilities: {prob_array}")
                #         print(f"Max prob: {max_probs[i]:.3f} at index {pred_idx}")
                #         print(f"Actual class: {results['actual'][i]}")
                
                # Get the raw predictions where confidence meets threshold
                raw_predictions = np.argmax(predictions_proba[confident_mask], axis=1)
                
                # Map predictions to correct classes
                predicted_classes[confident_mask] = raw_predictions
                
                # Filter results to only include confident predictions
                valid_indices = predicted_classes != -1
                filtered_actual = results['actual'][valid_indices]
                filtered_predicted = predicted_classes[valid_indices]
                
                if len(filtered_actual) > 0:
                    # Calculate confusion matrix for confident predictions
                    conf_matrix = confusion_matrix(filtered_actual, filtered_predicted, 
                                                labels=range(num_classes))
                    
                    # Calculate percentage matrix
                    conf_matrix_pct = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
                    conf_matrix_pct = np.nan_to_num(conf_matrix_pct)  # Handle division by zero
                    
                    # Plot heatmap
                    sns.heatmap(conf_matrix_pct, annot=conf_matrix, fmt='d', cmap='YlOrRd',
                            xticklabels=range(num_classes), yticklabels=range(num_classes), ax=ax)
                    
                    # Add dynamic directional indicators based on number of classes
                    negative_end = num_classes // 3
                    positive_start = num_classes - (num_classes // 3)
                    
                    # Add lines at positions that divide the classes into thirds
                    ax.axhline(y=negative_end, color='blue', linestyle='--', alpha=0.3)
                    ax.axhline(y=positive_start, color='blue', linestyle='--', alpha=0.3)
                    ax.axvline(x=negative_end, color='blue', linestyle='--', alpha=0.3)
                    ax.axvline(x=positive_start, color='blue', linestyle='--', alpha=0.3)
                    
                    # Add dynamically positioned direction labels
                    ax.text(-0.2, negative_end/2, 'Negative', rotation=90, verticalalignment='center')
                    ax.text(-0.2, (positive_start + negative_end)/2, 'Neutral', rotation=90, verticalalignment='center')
                    ax.text(-0.2, (positive_start + num_classes)/2, 'Positive', rotation=90, verticalalignment='center')
                    
                    # Calculate metrics
                    accuracy = (filtered_predicted == filtered_actual).mean()
                    
                    def get_direction(x):
                        negative_end = num_classes // 3
                        positive_start = num_classes - (num_classes // 3)
                        if x < negative_end: return 'negative'
                        elif x >= positive_start: return 'positive'
                        return 'neutral'
                    
                    pred_direction = np.vectorize(get_direction)(filtered_predicted)
                    actual_direction = np.vectorize(get_direction)(filtered_actual)
                    directional_accuracy = (pred_direction == actual_direction).mean()
                    
                    coverage = np.mean(confident_mask)
                    title = f'Threshold: {threshold}\n'
                    title += f'Coverage: {coverage:.2%}\n'
                    title += 'Color: % of True Class\nValues: Absolute Count'
                    ax.set_title(title)
                    ax.set_xlabel('Predicted Class')
                    ax.set_ylabel('True Class')
                    
                    ax.text(0, -0.2, f'Acc: {accuracy:.2f}\nDir Acc: {directional_accuracy:.2f}',
                            transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, f'No predictions meet\nthreshold {threshold}',
                        ha='center', va='center')
            
            plt.tight_layout()
            plt.show()

        # Replace the original confusion matrix plotting with:
        plot_threshold_confusion_matrices(
            results, 
            predictions_proba, 
            thresholds=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9],  # Configurable thresholds
            num_classes=len(model.classes_)
        )

        # Create visual confusion matrix
        conf_matrix = confusion_matrix(results['actual'], results['predicted'])
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages for each class
        conf_matrix_pct = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(conf_matrix_pct, annot=conf_matrix, fmt='d', cmap='YlOrRd',
                    xticklabels=range(num_classes), yticklabels=range(num_classes))
        
        # Add directional indicators
        plt.axhline(y=1.5, color='blue', linestyle='--', alpha=0.3)  # Separate negative classes
        plt.axhline(y=3.5, color='blue', linestyle='--', alpha=0.3)  # Separate positive classes
        plt.axvline(x=1.5, color='blue', linestyle='--', alpha=0.3)
        plt.axvline(x=3.5, color='blue', linestyle='--', alpha=0.3)
        
        plt.title('Confusion Matrix\nColor: % of True Class, Values: Absolute Count')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Add direction labels
        plt.text(-0.2, 0.5, 'Negative', rotation=90, verticalalignment='center')
        plt.text(-0.2, 2.5, 'Neutral', rotation=90, verticalalignment='center')
        plt.text(-0.2, 4, 'Positive', rotation=90, verticalalignment='center')
        
        plt.tight_layout()
        plt.plot()
        
        # Add average probability analysis
        print("\nAverage Prediction Probabilities:")
        avg_probs = pd.DataFrame(predictions_proba).mean()
        for i in range(len(avg_probs)):
            print(f"Class {i}: {avg_probs[i]:.3f}")
        
        # 2. Confidence Analysis
        def analyze_confidence_levels(results):
            # More granular confidence levels for detailed analysis
            confidence_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            stats = []
            
            # Get max probabilities for each prediction
            max_probs = predictions_proba.max(axis=1)
            
            # Print overall probability distribution stats
            print("\nProbability Distribution Statistics:")
            print(f"Mean max probability: {max_probs.mean():.3f}")
            print(f"Median max probability: {np.median(max_probs):.3f}")
            print(f"Std max probability: {max_probs.std():.3f}")
            print("\nMax probability percentiles:")
            for p in [10, 25, 50, 75, 90]:
                print(f"{p}th percentile: {np.percentile(max_probs, p):.3f}")
            
            for conf in confidence_levels:
                high_conf_mask = max_probs >= conf
                n_samples = high_conf_mask.sum()
                
                if n_samples > 0:
                    conf_accuracy = (results.loc[high_conf_mask, 'predicted'] == 
                                results.loc[high_conf_mask, 'actual']).mean()
                    conf_dir_accuracy = results.loc[high_conf_mask, 'direction_correct'].mean()
                    coverage = n_samples / len(results)
                    
                    # Get class distribution for high confidence predictions
                    high_conf_pred_dist = results.loc[high_conf_mask, 'predicted'].value_counts(normalize=True)
                    
                    stats.append({
                        'confidence_threshold': conf,
                        'accuracy': conf_accuracy,
                        'directional_accuracy': conf_dir_accuracy,
                        'coverage': coverage,
                        'n_samples': n_samples,
                        'most_common_class': high_conf_pred_dist.index[0] if len(high_conf_pred_dist) > 0 else None,
                        'most_common_class_freq': high_conf_pred_dist.iloc[0] if len(high_conf_pred_dist) > 0 else 0
                    })
            
            stats_df = pd.DataFrame(stats)
            print("\nDetailed Confidence Level Analysis:")
            print(stats_df.to_string(float_format=lambda x: '{:.3f}'.format(x) if isinstance(x, float) else str(x)))
            
            return stats_df
        
        conf_stats = analyze_confidence_levels(results)
        print("\nConfidence Level Analysis:")
        print(conf_stats)
        
        # 3. Visualization Functions
        def plot_directional_confidence(results):
            plt.figure(figsize=(15, 6))
            
            # Plot 1: Probability distributions for correct vs incorrect predictions
            plt.subplot(1, 2, 1)
            correct_mask = results['predicted'] == results['actual']
            
            sns.boxplot(data=pd.melt(results[['prob_negative', 'prob_neutral', 'prob_positive']], 
                                    var_name='direction', value_name='probability'))
            plt.title('Probability Distributions by Direction')
            plt.ylabel('Probability')
            
            # Plot 2: Directional accuracy over time
            plt.subplot(1, 2, 2)
            rolling_acc = results['direction_correct'].rolling(window=50).mean()
            plt.plot(rolling_acc.index, rolling_acc, label='50-period Rolling Directional Accuracy')
            plt.axhline(y=rolling_acc.mean(), color='r', linestyle='--', 
                    label='Average Directional Accuracy')
            plt.title('Directional Accuracy Over Time')
            plt.legend()
            
            plt.tight_layout()
            plt.plot()
        
        def plot_probability_heatmap(results):
            plt.figure(figsize=(12, 8))
            
            # Create probability matrix for heatmap
            avg_probs = np.zeros((num_classes, num_classes))
            
            # Convert predictions_proba to numpy array if it isn't already
            proba_array = np.array(predictions_proba)
            
            # Get numpy array of actual values
            actual_array = results['actual'].values
            
            for true_class in range(num_classes):
                class_indices = np.where(actual_array == true_class)[0]
                if len(class_indices) > 0:
                    avg_probs[true_class] = proba_array[class_indices].mean(axis=0)
            
            # Create the heatmap
            sns.heatmap(avg_probs, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                    vmin=0, vmax=1.0, center=0.5)
            plt.title('Average Prediction Probabilities by True Class')
            plt.xlabel('Predicted Class')
            plt.ylabel('True Class')
            
            plt.tight_layout()
            plt.plot()
        
        # 4. High Confidence Analysis
        def analyze_high_confidence_predictions(results, threshold=0.8):
            high_conf_mask = predictions_proba.max(axis=1) >= threshold
            high_conf_results = results[high_conf_mask]
            
            if len(high_conf_results) > 0:
                print(f"\nHigh Confidence Predictions (>{threshold}):")
                print(f"Count: {len(high_conf_results)}")
                print(f"Accuracy: {(high_conf_results['predicted'] == high_conf_results['actual']).mean():.2f}")
                print(f"Directional Accuracy: {high_conf_results['direction_correct'].mean():.2f}")
                
                # Analyze class distribution for high confidence predictions
                print("\nClass Distribution (High Confidence):")
                print(high_conf_results['predicted'].value_counts().sort_index())
        
        # Execute visualizations and analysis
        plot_directional_confidence(results)
        plot_probability_heatmap(results)
        analyze_high_confidence_predictions(results)
        
        # 5. Save detailed results for further analysis
        #results.to_csv(f'classifier_results_iter_{iteration_num}.csv')
        
        return results  # Return results DataFrame for potential further analysis

    def iteration_summary_regressor(self,results, model, iteration_num):
        if self.config.summary_per_iteration:
            Panel(
                histogram=[],
                right=[(results["close"], "close") if "close" in results.columns else ()],
                left=[],
                middle1=[(results["predicted"],"predicted"),(results["actual"],"actual")],
            ).chart(size="s", precision=6, title=f"Iteration {iteration_num}")            
            
            #calculate and plot directional accuracy
            def calculate_directional_accuracy(df, window=None):
                """
                Calculate directional accuracy between predicted and actual values.
                
                Parameters:
                -----------
                df : pandas.DataFrame
                    DataFrame with datetime index and columns 'predicted' and 'actual'
                window : int, optional
                    If provided, calculates rolling directional accuracy using this window size
                    
                Returns:
                --------
                dict
                    Dictionary containing accuracy metrics and optionally rolling accuracy series
                """
                # Calculate actual and predicted directions
                actual_direction = df['actual'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                predicted_direction = df['predicted'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                
                # Calculate correct predictions (excluding flat movements)
                correct_predictions = (actual_direction * predicted_direction == 1)
                total_movements = (actual_direction != 0) & (predicted_direction != 0)
                
                # Calculate overall accuracy
                overall_accuracy = (correct_predictions & total_movements).sum() / total_movements.sum()
                
                # Calculate direction-specific accuracy
                up_actual = actual_direction == 1
                down_actual = actual_direction == -1
                up_predicted = predicted_direction == 1
                down_predicted = predicted_direction == -1
                
                up_accuracy = (up_actual & up_predicted).sum() / up_actual.sum()
                down_accuracy = (down_actual & down_predicted).sum() / down_actual.sum()
                
                results = {
                    'overall_accuracy': overall_accuracy,
                    'up_accuracy': up_accuracy,
                    'down_accuracy': down_accuracy,
                    'total_predictions': total_movements.sum(),
                    'correct_predictions': (correct_predictions & total_movements).sum(),
                    'up_movements': up_actual.sum(),
                    'down_movements': down_actual.sum()
                }
                
                # If window is provided, calculate rolling accuracy
                if window:
                    # Overall rolling accuracy
                    rolling_correct = (correct_predictions & total_movements).rolling(window=window).sum()
                    rolling_total = total_movements.rolling(window=window).sum()
                    rolling_accuracy = rolling_correct / rolling_total
                    
                    # Direction-specific rolling accuracy
                    up_rolling_correct = (up_actual & up_predicted).rolling(window=window).sum()
                    up_rolling_total = up_actual.rolling(window=window).sum()
                    up_rolling_accuracy = up_rolling_correct / up_rolling_total
                    
                    down_rolling_correct = (down_actual & down_predicted).rolling(window=window).sum()
                    down_rolling_total = down_actual.rolling(window=window).sum()
                    down_rolling_accuracy = down_rolling_correct / down_rolling_total
                    
                    results.update({
                        'rolling_accuracy': rolling_accuracy,
                        'up_rolling_accuracy': up_rolling_accuracy,
                        'down_rolling_accuracy': down_rolling_accuracy
                    })
                
                return results

            def plot_directional_accuracy(df, results, window=None):
                """
                Create visualization of directional accuracy metrics.
                
                Parameters:
                -----------
                df : pandas.DataFrame
                    Original DataFrame with predictions
                results : dict
                    Results from calculate_directional_accuracy function
                window : int, optional
                    Window size used for rolling calculations
                """                
                # Create figure with subplots
                fig = plt.figure(figsize=(15, 10))
                gs = plt.GridSpec(2, 2, height_ratios=[2, 1])
                
                # Plot 1: Original Data and Predictions
                # ax1 = plt.subplot(gs[0, :])
                # ax1.plot(df.index, df['actual'], label='Actual', color='blue', alpha=0.7)
                # ax1.plot(df.index, df['predicted'], label='Predicted', color='red', alpha=0.7)
                # ax1.set_title('Actual vs Predicted Values')
                # ax1.legend()
                # ax1.grid(True)
                
                # Plot 2: Accuracy Metrics Bar Plot
                ax2 = plt.subplot(gs[1, 0])
                metrics = ['Overall', 'Up', 'Down']
                values = [results['overall_accuracy'], results['up_accuracy'], results['down_accuracy']]
                colors = ['blue', 'green', 'red']
                ax2.bar(metrics, values, color=colors, alpha=0.6)
                ax2.set_ylim(0, 1)
                ax2.set_title('Directional Accuracy by Type')
                ax2.set_ylabel('Accuracy')
                
                # Add percentage labels on bars
                for i, v in enumerate(values):
                    ax2.text(i, v + 0.01, f'{v:.1%}', ha='center')
                
                # Plot 3: Rolling Accuracy (if window provided)
                ax3 = plt.subplot(gs[1, 1])
                if window:
                    results['rolling_accuracy'].plot(ax=ax3, label='Overall', color='blue', alpha=0.7)
                    results['up_rolling_accuracy'].plot(ax=ax3, label='Up', color='green', alpha=0.7)
                    results['down_rolling_accuracy'].plot(ax=ax3, label='Down', color='red', alpha=0.7)
                    ax3.set_title(f'{window}-Period Rolling Accuracy')
                    ax3.set_ylim(0, 1)
                    ax3.legend()
                    ax3.grid(True)
                
                plt.tight_layout()
                return fig            

                # Calculate accuracy metrics with 30-day rolling window
            window = 30
            dir_acc_results = calculate_directional_accuracy(results, window=window)
            
            # Print summary statistics
            print("Directional Accuracy Metrics:")
            print(f"Overall Accuracy: {dir_acc_results['overall_accuracy']:.2%}")
            print(f"Up Movement Accuracy: {dir_acc_results['up_accuracy']:.2%}")
            print(f"Down Movement Accuracy: {dir_acc_results['down_accuracy']:.2%}")
            print(f"\nTotal Predictions: {dir_acc_results['total_predictions']}")
            print(f"Correct Predictions: {dir_acc_results['correct_predictions']}")
            print(f"Up Movements: {dir_acc_results['up_movements']}")
            print(f"Down Movements: {dir_acc_results['down_movements']}")
            
            # Create and display visualization
            fig = plot_directional_accuracy(results, dir_acc_results, window=window)
            plt.show()

            #actual vs predict distribution
            print(f"Actual: [{results['actual'].min():.2f}, {results['actual'].max():.2f}] | Predicted: [{results['predicted'].min():.2f}, {results['predicted'].max():.2f}]")

            fig = go.Figure()

            # Add both distributions
            fig.add_trace(go.Histogram(x=results['actual'], name='Actual', opacity=0.7, nbinsx=30))
            fig.add_trace(go.Histogram(x=results['predicted'], name='Predicted', opacity=0.7, nbinsx=30))

            # Update layout
            fig.update_layout(
                barmode='overlay',
                title='Distribution of Actual vs Predicted Values',
                xaxis_title='Value',
                yaxis_title='Count'
            )

            fig.show()

            # Calculate residuals and directions
            results['residuals'] = results['actual'] - results['predicted']
            results['direction'] = results['actual'].diff().apply(lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'Flat'))

            # Print overall and directional stats
            print(f"Overall residuals: [{results['residuals'].min():.2f}, {results['residuals'].max():.2f}], std: {results['residuals'].std():.2f}")
            print(f"Up moves residuals: mean={results[results['direction']=='Up']['residuals'].mean():.2f}, std={results[results['direction']=='Up']['residuals'].std():.2f}")
            print(f"Down moves residuals: mean={results[results['direction']=='Down']['residuals'].mean():.2f}, std={results[results['direction']=='Down']['residuals'].std():.2f}")

            # Create subplot with residual time series and histograms
            fig = sp.make_subplots(rows=2, cols=2, row_heights=[0.7, 0.3],
                                specs=[[{"colspan": 2}, None],
                                        [{}, {}]],
                                subplot_titles=('Residuals Over Time', 'Overall Distribution', 'Distribution by Direction'))

            # Add time series
            fig.add_trace(go.Scatter(x=results.index, y=results['residuals'], mode='lines', name='Residuals'), row=1, col=1)

            # Add overall histogram
            fig.add_trace(go.Histogram(x=results['residuals'], name='Overall', nbinsx=30), row=2, col=1)

            # Add directional histograms
            fig.add_trace(go.Histogram(x=results[results['direction']=='Up']['residuals'], name='Up Moves', nbinsx=30), row=2, col=2)
            fig.add_trace(go.Histogram(x=results[results['direction']=='Down']['residuals'], name='Down Moves', nbinsx=30), row=2, col=2)

            fig.update_layout(height=800, title='Residuals Analysis', barmode='overlay')
            fig.show()

            def plot_profits_analysis(results, threshold):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Count trades
                n_longs = (results['predicted'] > threshold).sum()
                n_shorts = (results['predicted'] < -threshold).sum()
                
                # Total profits breakdown
                profits = {
                    f'Total\n({n_longs + n_shorts} trades)': results['potential_profit'].sum(),
                    f'Long\n({n_longs} trades)': results.loc[results['predicted'] > threshold, 'potential_profit'].sum(),
                    f'Short\n({n_shorts} trades)': results.loc[results['predicted'] < -threshold, 'potential_profit'].sum()
                }
                ax1.bar(profits.keys(), profits.values())
                ax1.set_title('Total Profits Breakdown (Log Returns)')
                
                # Cumulative profits over time
                long_profit = results['potential_profit'].copy()
                short_profit = results['potential_profit'].copy()
                long_profit[results['predicted'] <= threshold] = 0
                short_profit[results['predicted'] >= -threshold] = 0
                
                results['potential_profit'].cumsum().plot(ax=ax2, label='Total', color='blue')
                long_profit.cumsum().plot(ax=ax2, label='Long', color='green')
                short_profit.cumsum().plot(ax=ax2, label='Short', color='red')
                
                ax2.set_title('Cumulative Log Returns Over Time')
                ax2.legend()
                
                plt.tight_layout()
                return fig

            def add_potential_profit(results, threshold, n_bars, slippage_pct=self.config.summary_slippage_pct):
                future_close = results['close'].shift(-n_bars)
                results['potential_profit'] = 0
                # Convert slippage from percentage to decimal
                slippage = slippage_pct / 100
                
                # For longs: buy at close*(1+slippage), sell at future_close*(1-slippage)
                results.loc[results['predicted'] > threshold, 'potential_profit'] = np.log(
                    (future_close*(1-slippage))/(results['close']*(1+slippage))
                )
                
                # For shorts: sell at close*(1-slippage), buy back at future_close*(1+slippage)
                results.loc[results['predicted'] < -threshold, 'potential_profit'] = np.log(
                    (results['close']*(1-slippage))/(future_close*(1+slippage))
                )

                plot_profits_analysis(results, threshold=threshold)  # or whatever threshold value you used
                plt.show()
                return results
            
            #display potential profit N bars in the future            
            results = add_potential_profit(results, self.config.summary_analysis_profit_th, self.config.forward_bars)

    def plot_feature_importance(self, model):
        # Get feature importance scores
        importance = pd.DataFrame({
            'feature': model.get_booster().feature_names,  # Assuming you have feature names list
            'importance': model.feature_importances_
        })

        # Sort by importance
        importance = importance.sort_values('importance', ascending=False)

        # Plot top 10 features
        plt.figure(figsize=(10, 6))
        plt.bar(importance['feature'][:30], importance['importance'][:30])
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        plt.show()


    def generate_feature_dataset(
            self,
            data: pd.DataFrame,
            output_path: Optional[str] = None,
            use_generic_features: bool = False,
            include_metadata: bool = False,
            num_iterations: Optional[int] = None
        ) -> pd.DataFrame:
        """
        Generate a dataset with features and targets using the same logic as run_rolling_window,
        processing train and test periods separately within each window.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with OHLCV columns
        output_path : str, optional
            Path to save the CSV file. If None, the dataset is only returned
        use_generic_features : bool
            If True, features will be renamed to feature_0, feature_1, etc.
        include_metadata : bool
            If True, includes 'period' and 'window' columns in the output
        num_iterations : int, optional
            Number of rolling window iterations to process. If None, process all possible windows
            
        Returns:
        --------
        pd.DataFrame
            Dataset containing all features and targets
        """
        # Get all date windows
        windows = self.get_date_windows(data)
        if num_iterations:
            windows = windows[:num_iterations]
        
        all_features_dfs = []
        warm_period = self.config.warm_up_period if self.config.warm_up_period is not None else 0
        
        print(f"Generating features dataset with {len(windows)} windows...")
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\nProcessing window {i+1}/{len(windows)}")
            
            # Handle warm-up period for both train and test data
            if warm_period > 0:
                train_warmup_data = data[data.index < train_start].tail(warm_period)
                train_start_with_warmup = train_warmup_data.index[0] if not train_warmup_data.empty else train_start
                
                test_warmup_data = data[data.index < test_start].tail(warm_period)
                test_start_with_warmup = test_warmup_data.index[0] if not test_warmup_data.empty else test_start
            else:
                train_start_with_warmup = train_start
                test_start_with_warmup = test_start
            
            # Get train and test data with warm-up periods
            train_mask = (data.index >= train_start_with_warmup) & (data.index < train_end)
            test_mask = (data.index >= test_start_with_warmup) & (data.index < test_end)
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            # Check for minimum required bars
            min_required_bars = max(20, self.config.forward_bars + 1)
            if len(train_data) < min_required_bars or len(test_data) < 1:
                print(f"Skipping window {i}: Insufficient data")
                continue
            
            try:
                # Generate features for train period
                train_features, feature_cols = self.feature_builder.prepare_features(train_data)
                train_target = self.feature_builder.create_target(train_features)
                
                # Generate features for test period
                test_features, _ = self.feature_builder.prepare_features(test_data)
                test_target = self.feature_builder.create_target(test_features, train_data=train_features)
                
                # Remove warmup period from features if it was used
                if warm_period > 0:
                    train_features = train_features[train_features.index >= train_start]
                    test_features = test_features[test_features.index >= test_start]
                    train_target = train_target[train_target.index >= train_start]
                    test_target = test_target[test_target.index >= test_start]
                
                # Combine features and targets
                train_features['target'] = train_target
                test_features['target'] = test_target
                
                # Add metadata if requested
                if include_metadata:
                    train_features['period'] = 'train'
                    test_features['period'] = 'test'
                    train_features['window'] = i
                    test_features['window'] = i
                
                # Combine train and test features
                window_features = pd.concat([train_features, test_features])
                
                # Remove NaN values and infinities
                window_features = window_features.replace([np.inf, -np.inf], np.nan)
                window_features = window_features.dropna()
                
                all_features_dfs.append(window_features)
                
            except Exception as e:
                print(f"Error processing window {i}: {str(e)}")
                continue
        
        if not all_features_dfs:
            raise ValueError("No valid features generated from any window")
        
        # Combine all windows
        final_dataset = pd.concat(all_features_dfs, axis=0)
        
        # Rename features if requested
        if use_generic_features:
            feature_columns = [col for col in final_dataset.columns 
                            if col not in ['target', 'period', 'window']]
            feature_mapping = {col: f'feature_{i}' for i, col 
                            in enumerate(feature_columns)}
            final_dataset = final_dataset.rename(columns=feature_mapping)
                
        # Save to CSV if output path is provided
        if output_path:
            print(f"\nSaving dataset to {output_path}")
            final_dataset.to_csv(output_path, index=True, index_label="Open time")
            print(f"Dataset saved successfully with {len(final_dataset)} rows and "
                f"{len(final_dataset.columns)} columns")
        
        return final_dataset