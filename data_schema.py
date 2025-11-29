from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class ColumnMeta:
    name: str
    symbol: str
    desc_zh: str
    unit: str
    is_measured: bool

COLUMN_SCHEMA: Dict[str, ColumnMeta] = {
    "t_s": ColumnMeta("t_s", "t", "时间", "s", False),
    "I_A": ColumnMeta("I_A", "I(t)", "电流(放电>0)", "A", True),
    "V_V": ColumnMeta("V_V", "V(t)", "端电压", "V", True),
    "Q_Ah": ColumnMeta("Q_Ah", "Q(t)", "累积电量", "Ah", False),
    "SOC": ColumnMeta("SOC", "SOC(t)", "荷电状态", "-", False),
    "T_amb_degC": ColumnMeta("T_amb_degC", "T_amb(t)", "环境温度", "°C", True),
    "T_surf_degC": ColumnMeta("T_surf_degC", "T_surf(t)", "壳体表面温度", "°C", True),
    "eps_eq": ColumnMeta("eps_eq", "ε_eq(t)", "壳体等效应变", "-", True),

    # --- 关键修改：区分测量值与真值 ---
    "T_in_degC": ColumnMeta("T_in_degC", "T_in_meas(t)", "内部温度(标定测量/含噪)", "°C", True),
    "P_gas_kPa": ColumnMeta("P_gas_kPa", "P_gas_meas(t)", "内部气体压力(标定测量/含噪)", "kPa", True),

    "T_core_phys_degC": ColumnMeta("T_core_phys_degC", "T_core_true(t)", "内部核心温度(物理真值)", "°C", False),
    "P_gas_phys_kPa": ColumnMeta("P_gas_phys_kPa", "P_gas_true(t)", "内部气体压力(物理真值)", "kPa", False),
    # ------------------------------------

    "cell_id": ColumnMeta("cell_id", "-", "电池编号", "-", False),
    "cycle_index": ColumnMeta("cycle_index", "-", "循环编号", "-", False),
    "step_index": ColumnMeta("step_index", "-", "工况步骤编号", "-", False),
    "is_calib": ColumnMeta("is_calib", "-", "是否标定电池", "-", False),
    "state_tag": ColumnMeta("state_tag", "-", "状态标签", "-", False),
    "fault_type": ColumnMeta("fault_type", "-", "故障类型", "-", False), # 新增
    "scenario": ColumnMeta("scenario", "-", "场景名称", "-", False), # 新增
    "is_anomaly": ColumnMeta("is_anomaly", "-", "是否异常", "-", False), # 新增
}

# (generate_virtual_battery_data 函数此处省略，V3版不再依赖它，直接用 synthetic_data)