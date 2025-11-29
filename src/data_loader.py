"""
src/data_loader.py (V1.1 Enhanced)
负责将外部原始数据(CSV/Excel)转换为项目标准DataFrame格式。
增强点：schema校验、时间轴去重排序、更健壮的重采样。
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .data_schema import COLUMN_SCHEMA

def load_experiment_data(
    file_path: str,
    channel_map: dict[str, str],
    unit_factors: dict[str, float] = None,
    dt_target: float = 1.0,
    time_col: str = None
) -> pd.DataFrame:
    
    # 1. 读取文件
    if file_path.endswith('.csv'):
        df_raw = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df_raw = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")

    # 2. 列名映射
    df = df_raw.rename(columns=channel_map)
    
    # 3. 单位换算
    if unit_factors:
        for col, factor in unit_factors.items():
            if col in df.columns:
                df[col] = df[col] * factor

    # 4. 时间轴处理
    if time_col and time_col in df_raw.columns:
        t_raw = df_raw[time_col]
        # 判断是否为datetime类型
        if pd.api.types.is_datetime64_any_dtype(t_raw) or isinstance(t_raw.iloc[0], str):
            t_raw = pd.to_datetime(t_raw)
            t_seconds = (t_raw - t_raw.iloc[0]).dt.total_seconds()
        else:
            # 假设数值即秒
            t_seconds = t_raw.astype(float) - float(t_raw.iloc[0])
        df['t_s'] = t_seconds
    elif 't_s' not in df.columns:
        # 无时间列，默认等间隔
        df['t_s'] = np.arange(len(df), dtype=float) * dt_target

    # 5. [增强] 时间轴清洗
    df = df.sort_values('t_s').drop_duplicates('t_s', keep='first')
    df = df.set_index('t_s')

    # 6. [增强] 鲁棒重采样
    t_min, t_max = float(df.index.min()), float(df.index.max())
    new_index = np.arange(t_min, t_max + dt_target, dt_target)
    
    # 联合索引 -> 插值 -> 重索引
    df_resampled = (
        df.reindex(df.index.union(new_index))
          .interpolate(method='linear')
          .reindex(new_index)
    )
    df_resampled.index.name = 't_s'
    
    # 7. [增强] Schema 校验
    df_resampled.attrs['dt_s'] = dt_target
    
    required_cols = ['I_A', 'V_V', 'T_surf_degC']
    missing = [c for c in required_cols if c not in df_resampled.columns]
    if missing:
        # 实际工程中可能允许缺失某些列，这里仅打印警告
        print(f"[Warn] Missing standard columns: {missing}")
        
    return df_resampled