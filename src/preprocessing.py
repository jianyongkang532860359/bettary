"""
src/preprocessing.py (V1.2 Robust)
信号预处理模块：滤波、异常值标记与修复。
增强点：
1. 修复 Pandas Deprecation Warning (.ffill() / .bfill())
2. 增强滤波器鲁棒性 (长度检查 + Nyquist检查)
3. 异常值处理逻辑优化
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
    lowpass_cutoff: float = 0.5  # Hz
    outlier_cols: List[str] = field(default_factory=lambda: ['I_A', 'V_V', 'eps_eq'])
    outlier_sigma: float = 5.0

class SignalPreprocessor:
    def __init__(self, config: PreprocessConfig = None):
        self.config = config if config else PreprocessConfig()
        self._dt = 1.0

    def fit(self, df: pd.DataFrame):
        """
        从数据中推断采样间隔 dt
        """
        if 'dt_s' in df.attrs:
            self._dt = df.attrs['dt_s']
        else:
            t = df.index.values
            if len(t) > 1:
                # 使用中位数以抵抗时间戳抖动
                dt_est = np.median(np.diff(t))
                self._dt = float(dt_est) if dt_est > 0 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        
        # 安全计算采样率，防止除零
        fs = 1.0 / self._dt if self._dt > 1e-6 else 1.0
        
        # 1. 低通滤波
        for col in self.config.lowpass_cols:
            if col in df_out.columns:
                # [Fix]: 替换 deprecated method='ffill'
                data_series = df_out[col].ffill().bfill()
                
                # 如果整列都是 NaN，跳过
                if data_series.isna().all():
                    continue
                    
                data = data_series.values
                
                # 检查数据长度 (2阶滤波器需要至少 9-12 个点)
                if len(data) < 15:
                    continue
                
                # Nyquist 频率检查 (截止频率必须 < fs/2)
                nyq = 0.5 * fs
                cutoff = min(self.config.lowpass_cutoff, nyq * 0.95) # 留 5% 裕量
                
                try:
                    b, a = butter(2, cutoff / nyq, btype='low')
                    df_out[col] = filtfilt(b, a, data)
                except Exception as e:
                    print(f"[Preprocess Warn] Filter failed on {col}: {e}")

        # 2. 异常值处理 (Robust Z-Score)
        for col in self.config.outlier_cols:
            if col in df_out.columns:
                data = df_out[col].values
                median = np.nanmedian(data)
                diff = np.abs(data - median)
                mad = np.nanmedian(diff)
                
                if mad > 1e-9:
                    z_score = 0.6745 * diff / mad
                    mask = z_score > self.config.outlier_sigma
                    
                    if np.any(mask):
                        flag_col = f"{col}_is_outlier"
                        # 标记异常点 (0/1)
                        df_out[flag_col] = 0
                        df_out.loc[mask, flag_col] = 1
                        
                        # 将异常值设为 NaN，然后插值修复
                        df_out.loc[mask, col] = np.nan
                        df_out[col] = df_out[col].interpolate(method='linear', limit_direction='both')
        
        return df_out