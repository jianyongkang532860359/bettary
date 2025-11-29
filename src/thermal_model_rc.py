from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd

try:
    from scipy import signal
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

@dataclass
class RCThermalParams:
    C_s: float   # J/K
    C_m: float   # J/K
    C_c: float   # J/K
    R_sa: float  # K/W
    R_sm: float  # K/W
    R_mc: float  # K/W
    k_q: float   # W/A^2

def get_continuous_ss(params: RCThermalParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = params
    # 3-Node RC matrices: State x=[Ts, Tm, Tc], Input u=[Tamb, Pheat]
    A = np.array([
        [-(1/p.R_sa + 1/p.R_sm)/p.C_s, (1/p.R_sm)/p.C_s, 0],
        [(1/p.R_sm)/p.C_m, -(1/p.R_sm + 1/p.R_mc)/p.C_m, (1/p.R_mc)/p.C_m],
        [0, (1/p.R_mc)/p.C_c, -(1/p.R_mc)/p.C_c]
    ])
    B = np.zeros((3, 2))
    B[0, 0] = 1.0 / (p.C_s * p.R_sa)
    B[2, 1] = 1.0 / p.C_c 
    C = np.eye(3)
    D = np.zeros((3, 2))
    return A, B, C, D

def simulate_rc_thermal_fast(
    params: RCThermalParams,
    T_amb_degC: np.ndarray,
    dt_s: float,
    P_heat_W: Optional[np.ndarray] = None,
    I_A: Optional[np.ndarray] = None,
    T_init: Optional[Union[np.ndarray, float]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    T_amb = np.asarray(T_amb_degC, dtype=float)
    if P_heat_W is None:
        if I_A is None: raise ValueError("Need P_heat_W or I_A")
        P_heat = (np.asarray(I_A)**2) * params.k_q
    else:
        P_heat = np.asarray(P_heat_W, dtype=float)

    if not _HAS_SCIPY:
        raise ImportError("scipy is required for fast simulation.")

    # LTI Simulation
    A, B, C, D = get_continuous_ss(params)
    sys_d = (np.eye(3) + A * dt_s, B * dt_s, C, D, dt_s)
    
    if T_init is None:
        t0 = T_amb[0]
        x0 = np.array([t0, t0, t0])
    elif isinstance(T_init, (float, int)):
        x0 = np.array([float(T_init)]*3)
    else:
        x0 = np.asarray(T_init)

    u = np.column_stack((T_amb, P_heat))
    _, y_out, _ = signal.dlsim(sys_d, u, x0=x0)
    
    return y_out[:, 0], y_out[:, 1], y_out[:, 2] # Ts, Tm, Tc

def simulate_rc_from_df(
    df: pd.DataFrame,
    params: RCThermalParams,
    dt_s: Optional[float] = None,
) -> pd.DataFrame:
    if dt_s is None:
        dt_s = df.attrs.get("dt_s", 1.0)
    
    T_init = None
    if "T_surf_degC" in df.columns and np.isfinite(df["T_surf_degC"].iloc[0]):
        T_init = float(df["T_surf_degC"].iloc[0])

    T_s, T_m, T_c = simulate_rc_thermal_fast(
        params, 
        df["T_amb_degC"].values, 
        dt_s, 
        I_A=df["I_A"].values,
        T_init=T_init
    )
    
    df_out = df.copy()
    df_out["T_surf_model_degC"] = T_s
    df_out["T_core_phys_degC"] = T_c
    return df_out