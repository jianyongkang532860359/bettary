"""
src/preprocessing.py (V1.1 Enhanced)
信号预处理模块：滤波、异常值标记与修复。
增强点：
1. 异常值处理保留 flag 列 (is_outlier)。
2. 滤波前的长度检查。
3. Fit 接口增强 (预留基线学习能力)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PreprocessConfig:
    lowpass_cols: List[str] = field(default_factory=lambda: ['eps_eq', 'T_surf_degC'])
    lowpass_cutoff: float = 0.5
    outlier_cols: List[str] = field(default_factory=lambda: ['I_A', 'V_V', 'eps_eq'])
    outlier_sigma: float = 5.0
    # 未来可扩展: enable_temp_drift: bool = True

class SignalPreprocessor:
    def __init__(self, config: PreprocessConfig = None):
        self.config = config if config else PreprocessConfig()
        self._dt = 1.0
        self._state = {} # 用于存储 fit 学到的基线参数

    def fit(self, df: pd.DataFrame):
        """
        从数据中学习参数 (如 dt_s, 温漂基线等)
        """
        if 'dt_s' in df.attrs:
            self._dt = df.attrs['dt_s']
        else:
            t = df.index.values
            if len(t) > 1:
                self._dt = float(np.median(np.diff(t)))
        
        # [Future] 这里可以拟合 alpha_th_pre 等基线参数
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用预处理：低通滤波 -> 异常值标记与修复
        """
        df_out = df.copy()
        fs = 1.0 / self._dt if self._dt > 0 else 1.0
        
        # 1. 低通滤波
        for col in self.config.lowpass_cols:
            if col in df_out.columns:
                data = df_out[col].fillna(method='ffill').fillna(method='bfill').values
                
                # [增强] 长度检查
                if len(data) < 15: # 滤波器阶数 padlen 保护
                    print(f"[Warn] Data too short for filter: {col}")
                    continue
                    
                fc = min(self.config.lowpass_cutoff, fs/2.1) 
                b, a = butter(2, fc / (fs/2), btype='low')
                try:
                    df_out[col] = filtfilt(b, a, data)
                except Exception as e:
                    print(f"[Warn] Filter failed on {col}: {e}")

        # 2. 异常值处理 (Robust Z-Score + Flagging)
        for col in self.config.outlier_cols:
            if col in df_out.columns:
                data = df_out[col].values
                median = np.nanmedian(data)
                mad = np.nanmedian(np.abs(data - median))
                
                if mad > 1e-9:
                    z_score = 0.6745 * (data - median) / mad
                    mask = np.abs(z_score) > self.config.outlier_sigma
                    
                    if np.any(mask):
                        # [增强] 保留证据：标记 Outlier
                        flag_col = f"{col}_is_outlier"
                        df_out[flag_col] = 0
                        df_out.loc[mask, flag_col] = 1
                        
                        # 修复数据 (以便后续计算)
                        df_out.loc[mask, col] = np.nan
                        df_out[col] = df_out[col].interpolate()
        
        return df_out