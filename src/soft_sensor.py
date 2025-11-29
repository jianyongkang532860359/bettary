from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings

# 尝试导入 ML 库
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

@dataclass
class ModelConfig:
    window_sizes: List[int] = field(default_factory=lambda: [10, 30])
    use_derivatives: bool = True
    target_cols: List[str] = field(default_factory=lambda: ['T_core_phys_degC', 'P_gas_phys_kPa'])
    feature_cols: List[str] = field(default_factory=lambda: [
        'I_A', 'V_V', 'T_surf_degC', 'T_amb_degC', 'eps_eq', 'SOC'
    ])
    model_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    })

def split_by_group(
    df: pd.DataFrame,
    group_col: str = "scenario",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if group_col not in df.columns:
        if 'cycle_index' in df.columns:
            group_col = 'cycle_index'
        else:
            n = int(len(df) * (1 - test_ratio))
            return df.iloc[:n], df.iloc[n:]
            
    groups = df[group_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    
    n_test = max(1, int(len(groups) * test_ratio))
    test_groups = groups[:n_test]
    
    df_test = df[df[group_col].isin(test_groups)].copy()
    df_train = df[~df[group_col].isin(test_groups)].copy()
    
    return df_train, df_test

class DataDrivenEstimator:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config if config else ModelConfig()
        self.model = None
        self._is_fitted = False
        self.feature_names_in_: List[str] = []
        
        if not _HAS_ML:
            warnings.warn("sklearn/xgboost not found.")

    def _feat_eng_group_logic(self, sub_df: pd.DataFrame) -> pd.DataFrame:
        df_feat = sub_df[self.config.feature_cols].copy()
        if 'T_surf_degC' in df_feat and 'T_amb_degC' in df_feat:
            df_feat['dT_surf_amb'] = df_feat['T_surf_degC'] - df_feat['T_amb_degC']
            
        cols_to_process = [c for c in ['I_A', 'V_V', 'T_surf_degC', 'eps_eq'] if c in df_feat.columns]
        
        if self.config.use_derivatives:
            for c in cols_to_process:
                df_feat[f'd{c}'] = df_feat[c].diff()
        
        for w in self.config.window_sizes:
            for c in cols_to_process:
                r = df_feat[c].rolling(window=w, min_periods=1)
                df_feat[f'{c}_mean_{w}'] = r.mean()
                df_feat[f'{c}_std_{w}'] = r.std()
                
        return df_feat

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        group_keys = []
        if 'cell_id' in df.columns: group_keys.append('cell_id')
        if 'cycle_index' in df.columns: group_keys.append('cycle_index')
        
        if group_keys:
            X = df.groupby(group_keys, group_keys=False).apply(self._feat_eng_group_logic)
        else:
            X = self._feat_eng_group_logic(df)
            
        # NaN 处理: ffill -> bfill -> 0.0
        X = X.ffill().bfill().fillna(0.0)
        return X.select_dtypes(include=[np.number])

    def fit(self, df_train: pd.DataFrame):
        if not _HAS_ML: raise ImportError("ML libraries missing.")
        
        targets = self.config.target_cols
        df_valid = df_train.dropna(subset=targets).copy()
        
        if len(df_valid) < 50:
            raise ValueError("Not enough valid training data.")
            
        X = self._feature_engineering(df_valid)
        self.feature_names_in_ = X.columns.tolist()
        y = df_valid[targets]
        
        est = XGBRegressor(**self.config.model_params)
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', MultiOutputRegressor(est))
        ])
        
        print(f"Training SoftSensor on {len(X)} samples, {len(self.feature_names_in_)} features...")
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted: raise RuntimeError("Model not fitted.")
        
        X = self._feature_engineering(df)
        X = X.reindex(columns=self.feature_names_in_, fill_value=0.0)
        
        y_pred = self.model.predict(X)
        
        df_out = df.copy()
        for i, col in enumerate(self.config.target_cols):
            df_out[f"{col}_pred"] = y_pred[:, i]
            
        return df_out

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, float]:
        targets = self.config.target_cols
        df_valid = df_test.dropna(subset=targets)
        if len(df_valid) == 0: return {}
        
        df_pred = self.predict(df_valid)
        metrics = {}
        
        for col in targets:
            y_true = df_valid[col]
            y_pred = df_pred[f"{col}_pred"]
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            metrics[f"{col}_MAE"] = mae
            metrics[f"{col}_RMSE"] = rmse
            metrics[f"{col}_R2"] = r2
            
        # [关键修复] 确保这一行存在且没有被缩进错误
        return metrics