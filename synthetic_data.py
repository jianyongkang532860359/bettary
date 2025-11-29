from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import pandas as pd
from .thermal_model_rc import RCThermalParams, simulate_rc_thermal_fast
from .mech_strain_pressure import MechParams

@dataclass
class NoiseConfig:
    sigma_Tsurf: float = 0.05
    sigma_Tin: float = 0.1
    sigma_eps: float = 1e-6
    sigma_V: float = 0.005
    sigma_I: float = 0.02
    sigma_Pgas: float = 0.5
    outlier_prob: float = 0.001
    missing_prob: float = 0.001
    def __post_init__(self):
        if not (0 <= self.outlier_prob <= 1): raise ValueError("probs must be in [0,1]")
        if not (0 <= self.missing_prob <= 1): raise ValueError("probs must be in [0,1]")

@dataclass
class SyntheticScenarioConfig:
    name: str
    duration_s: float
    dt_s: float = 1.0
    profile: str = "cc_discharge"
    I_nominal_A: float = 5.0
    T_amb_base_degC: float = 25.0
    anomaly_type: str = "none"   # none, overpressure, sensor_fault
    anomaly_level: str = "none"  # mild, severe
    is_calib: bool = True

def _solve_strain_from_pressure(P_target: np.ndarray, mech_params: MechParams, model_type: str) -> np.ndarray:
    k1, k3, p0 = mech_params.k1, mech_params.k3, mech_params.p0
    P_net = np.maximum(P_target - p0, 0.0)
    
    if model_type == 'linear':
        k1_safe = k1 if k1 > 1e-3 else 1e-3
        return P_net / k1_safe
    else:
        eps_grid = np.linspace(0, 0.05, 5000)
        P_grid = k1 * eps_grid + k3 * (eps_grid**3)
        P_clamped = np.clip(P_net, P_grid.min(), P_grid.max())
        return np.interp(P_clamped, P_grid, eps_grid)

def _add_noise_and_defects(rng, x_true, sigma, noise_cfg):
    n = len(x_true)
    x_meas = x_true + rng.normal(0, sigma, n)
    if noise_cfg.outlier_prob > 0:
        mask_o = rng.random(n) < noise_cfg.outlier_prob
        if np.any(mask_o):
            std_val = np.nanstd(x_true) + 1e-6
            x_meas[mask_o] += rng.normal(0, 5.0 * std_val, size=mask_o.sum())
    if noise_cfg.missing_prob > 0:
        mask_m = rng.random(n) < noise_cfg.missing_prob
        if np.any(mask_m):
            x_meas[mask_m] = np.nan
    return x_meas

def generate_synthetic_cycle(
    scenario: SyntheticScenarioConfig,
    rc_params: RCThermalParams,
    mech_params: MechParams,
    mech_model_type: str,
    capacity_Ah: float,
    cell_id: str,
    cycle_index: int,
    noise: NoiseConfig,
    seed: int
) -> pd.DataFrame:
    
    rng = np.random.default_rng(seed)
    n = int(scenario.duration_s / scenario.dt_s) + 1
    t = np.arange(0, n * scenario.dt_s, scenario.dt_s)[:n]
    
    # 1. 基础工况
    I_true = np.full_like(t, scenario.I_nominal_A, dtype=float)
    if scenario.profile == "dynamic": I_true += 2.0 * np.sin(t / 100.0)
    I_true += rng.normal(0, 0.01, n)
    T_amb_true = np.full_like(t, scenario.T_amb_base_degC, dtype=float) + rng.normal(0, 0.01, n)
    
    Q = np.cumsum(I_true * scenario.dt_s / 3600.0)
    SOC = np.clip(1.0 - Q / capacity_Ah, 0.0, 1.0)
    
    # 3. RC 热模型
    T_s_true, _, T_c_true = simulate_rc_thermal_fast(rc_params, T_amb_true, scenario.dt_s, I_A=I_true, T_init=T_amb_true[0])
    
    # 4. 力学模型 (物理基准)
    eps_base = 500e-6 * (1.0 - SOC) + 10e-6 * (T_c_true - mech_params.t_ref)
    if mech_model_type == 'linear':
        P_base = mech_params.k1 * eps_base + mech_params.p0
    else:
        P_base = mech_params.k1 * eps_base + mech_params.k3 * (eps_base**3) + mech_params.p0
        
    # 5. 异常注入 (True Physics)
    P_true = P_base.copy()
    
    if scenario.anomaly_type == 'overpressure':
        # 真故障：物理压力 P_true 升高
        if scenario.anomaly_level == 'mild':
            mask = SOC < 0.5
            if np.any(mask): P_true[mask] += 20.0 * (1.0 - SOC[mask] / 0.5)
        elif scenario.anomaly_level == 'severe':
            step_idx = int(0.8 * n)
            P_true[step_idx:] += 80.0
        # 物理反演：应变随之升高
        eps_true = _solve_strain_from_pressure(P_true, mech_params, mech_model_type)
    else:
        # 正常 或 传感器故障：物理压力 P_true 保持基准，应变保持基准
        # (sensor_fault 只影响测量层，不影响真值层)
        eps_true = eps_base

    # 7. 观测值生成 (含噪声)
    I_meas = _add_noise_and_defects(rng, I_true, noise.sigma_I, noise)
    V_true = (3.0 + 1.2 * SOC) - 0.02 * I_true
    V_meas = _add_noise_and_defects(rng, V_true, noise.sigma_V, noise)
    T_surf_meas = _add_noise_and_defects(rng, T_s_true, noise.sigma_Tsurf, noise)
    T_amb_meas = _add_noise_and_defects(rng, T_amb_true, noise.sigma_Tsurf, noise)
    eps_meas = _add_noise_and_defects(rng, eps_true, noise.sigma_eps, noise)
    
    T_in_meas = np.full(n, np.nan)
    P_gas_meas = np.full(n, np.nan)
    
    if scenario.is_calib:
        T_in_meas = _add_noise_and_defects(rng, T_c_true, noise.sigma_Tin, noise)
        
        # [核心修正]: Sensor Fault 注入点
        # 仅修改传入 add_noise 的基准值，P_true (真值) 保持不变
        P_meas_base = P_true.copy() 
        if scenario.anomaly_type == 'sensor_fault':
             step_idx = int(0.8 * n)
             bias = 20.0 if scenario.anomaly_level == 'mild' else 50.0
             P_meas_base[step_idx:] += bias # 模拟传感器漂移
             
        P_gas_meas = _add_noise_and_defects(rng, P_meas_base, noise.sigma_Pgas, noise)

    df = pd.DataFrame({
        "t_s": t, "I_A": I_meas, "V_V": V_meas, "Q_Ah": Q, "SOC": SOC,
        "T_surf_degC": T_surf_meas, "T_amb_degC": T_amb_meas, "eps_eq": eps_meas,
        "T_in_degC": T_in_meas, "P_gas_kPa": P_gas_meas, # Meas
        "T_core_phys_degC": T_c_true, "P_gas_phys_kPa": P_true, # True
        "cell_id": cell_id, "cycle_index": cycle_index, "scenario": scenario.name,
        "is_calib": scenario.is_calib, "is_anomaly": scenario.anomaly_type != 'none',
        "fault_type": scenario.anomaly_type
    }).set_index("t_s")
    df.attrs["dt_s"] = scenario.dt_s
    return df

def generate_synthetic_dataset(
    scenarios: List[SyntheticScenarioConfig],
    rc_params: RCThermalParams,
    mech_params: MechParams,
    mech_model_type: str,
    capacity_Ah: float,
    noise: NoiseConfig,
    base_seed: int = 42
) -> pd.DataFrame:
    dfs = []
    for i, sc in enumerate(scenarios):
        this_mech = mech_params
        if not sc.is_calib:
            rng_p = np.random.default_rng(base_seed + i)
            this_mech = MechParams(
                k1=mech_params.k1 * rng_p.uniform(0.95, 1.05),
                k3=mech_params.k3, p0=mech_params.p0, alpha_th=mech_params.alpha_th
            )
        df = generate_synthetic_cycle(
            sc, rc_params, this_mech, mech_model_type, capacity_Ah,
            f"cell_{i}", i, noise, base_seed + i
        )
        dfs.append(df)
    return pd.concat(dfs)