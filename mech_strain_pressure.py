from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass

@dataclass
class MechParams:
    k1: float = 0.0
    k3: float = 0.0
    p0: float = 0.0
    alpha_th: float = 0.0
    t_ref: float = 25.0

class MechanicalPressureModel:
    def __init__(self, model_type: str = "cubic", alpha_thermal: float = 0.0):
        self.model_type = model_type
        self.params = MechParams(alpha_th=alpha_thermal)
        self._is_fitted = False

    def _get_mechanical_strain(self, eps_meas, t_surf):
        if self.params.alpha_th == 0.0 or t_surf is None:
            return np.asarray(eps_meas)
        dt = np.asarray(t_surf) - self.params.t_ref
        return np.asarray(eps_meas) - self.params.alpha_th * dt

    def fit(self, df: pd.DataFrame, min_points=30, eps_range_min=1e-6):
        df_v = df.dropna(subset=["eps_eq", "P_gas_kPa", "T_surf_degC"])
        if len(df_v) < min_points: raise ValueError("Not enough data to fit.")
        
        self.params.t_ref = float(df_v["T_surf_degC"].mean())
        eps_mech = self._get_mechanical_strain(df_v["eps_eq"], df_v["T_surf_degC"])
        
        if np.ptp(eps_mech) < eps_range_min:
            raise ValueError("Mechanical strain variation too small.")

        p_true = df_v["P_gas_kPa"].values
        
        if self.model_type == 'linear':
            popt, _ = curve_fit(lambda x,k1,p0: k1*x+p0, eps_mech, p_true, 
                                p0=[1e5, 100], bounds=([0,0], [np.inf, np.inf]))
            self.params.k1, self.params.p0 = popt
        else:
            popt, _ = curve_fit(lambda x,k1,k3,p0: k1*x+k3*x**3+p0, eps_mech, p_true, 
                                p0=[1e5, 1e8, 100], bounds=([0,0,0], [np.inf, np.inf, np.inf]))
            self.params.k1, self.params.k3, self.params.p0 = popt
        self._is_fitted = True

    def predict(self, df: pd.DataFrame, out_col: str = "P_gas_mech_est") -> pd.DataFrame:
        """
        默认输出列名为 'P_gas_mech_est'，避免覆盖真值 'P_gas_phys_kPa'。
        """
        if not self._is_fitted: raise RuntimeError("Model not fitted.")
        t_surf = df["T_surf_degC"].values if "T_surf_degC" in df.columns else None
        eps_mech = self._get_mechanical_strain(df["eps_eq"], t_surf)
        
        mode = self.model_type
        k1, k3, p0 = self.params.k1, self.params.k3, self.params.p0
        if mode == 'linear': p_pred = k1 * eps_mech + p0
        else: p_pred = k1 * eps_mech + k3 * (eps_mech**3) + p0
        
        df_out = df.copy()
        df_out[out_col] = p_pred
        return df_out