"""
realtime_monitor.py
[BMS HIL Dashboard] - 实时在线监控上位机 (V14.3 Fix: Added deque import)
"""
import time
import queue
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque  # [Fix]: 补全这个引用

# ==========================================
# Import Core Modules
# ==========================================
from src.mech_strain_pressure import MechanicalPressureModel, MechParams
from src.thermal_model_rc import RCThermalParams
from src.soft_sensor import DataDrivenEstimator, ModelConfig
from src.synthetic_data import NoiseConfig, SyntheticScenarioConfig, generate_synthetic_dataset
from src.online_engine import OnlineBMSEngine

class DataStreamer:
    """Simulates real-time sensor data stream from DataFrame."""
    def __init__(self, df, speed_hz=10):
        self.df = df
        self.speed = speed_hz
        self.queue = queue.Queue()
        self.running = False
        
    def start(self):
        self.running = True
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        
    def _worker(self):
        print(">> [Streamer] Sensor started.")
        while self.running:
            for idx, row in self.df.iterrows():
                if not self.running: break
                
                sample = row.to_dict()
                # Fix: Ensure t_s is in dict
                if 't_s' not in sample: sample['t_s'] = idx 
                
                self.queue.put(sample)
                time.sleep(1.0 / self.speed)
            
            print(">> [Streamer] Loop reset.")
            time.sleep(1)

def setup_system():
    print(">> [Init] Pre-training models...")
    # Configs
    rc_p = RCThermalParams(800, 300, 1000, 2.0, 1.0, 0.5, 0.5)
    mech_p = MechParams(2e5, 1e8, 100, 23e-6)
    
    # 1. Train on Healthy Data
    scens = [SyntheticScenarioConfig("calib_golden", 1800, is_calib=True)]
    df_train = generate_synthetic_dataset(scens, rc_p, mech_p, "cubic", 10.0, NoiseConfig())
    df_train = df_train[df_train["fault_type"]=="none"]
    
    # D2
    mech = MechanicalPressureModel("cubic", alpha_thermal=23e-6)
    mech.fit(df_train)
    
    # D3 (Hybrid features need to be consistent with OnlineEngine)
    est = DataDrivenEstimator(ModelConfig(target_cols=["P_gas_phys_kPa", "T_core_phys_degC"], 
                                          window_sizes=[10, 60])) 
    est.fit(df_train)
    
    # 2. Generate Test Stream (Fault + Aging)
    print(">> [Init] Generating Test Stream...")
    scen_fault = SyntheticScenarioConfig("fleet_fault_op", 1800, is_calib=False, 
                                         anomaly_type="overpressure", anomaly_level="severe")
    scen_aging = SyntheticScenarioConfig("fleet_aging_test", 1800, is_calib=False)
    
    df_test = generate_synthetic_dataset([scen_fault, scen_aging], rc_p, mech_p, "cubic", 10.0, NoiseConfig(outlier_prob=0.0))
    
    # Inject Aging Physics (Current scaling)
    mask = df_test["scenario"] == "fleet_aging_test"
    df_test.loc[mask, "I_A"] *= 0.9
    
    return mech, est, df_test

def main():
    mech, soft, df_stream = setup_system()
    
    # Initialize Engine
    engine = OnlineBMSEngine(mech, soft, th_phys=10.0, th_meas=10.0)
    
    # Start Data Stream (50Hz Fast Forward)
    streamer = DataStreamer(df_stream, speed_hz=50)
    streamer.start()
    
    # Setup Plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 3, figure=fig)
    
    ax_p = fig.add_subplot(gs[0, :2])
    ax_t = fig.add_subplot(gs[1, :2])
    ax_risk = fig.add_subplot(gs[2, :2])
    ax_soc = fig.add_subplot(gs[0, 2])
    ax_soh = fig.add_subplot(gs[1, 2])
    ax_info = fig.add_subplot(gs[2, 2]); ax_info.axis('off')
    
    # Lines
    ln_p_mech, = ax_p.plot([], [], 'g--', label='Mech')
    ln_p_soft, = ax_p.plot([], [], 'r:', label='Soft')
    ln_p_meas, = ax_p.plot([], [], 'c.', ms=2, label='Meas')
    ax_p.set_ylabel('Press (kPa)'); ax_p.legend(loc=2); ax_p.grid(alpha=0.2)
    ax_p.set_ylim(80, 300)
    
    ln_t_core, = ax_t.plot([], [], 'm-', label='Core T (AI)')
    ln_t_surf, = ax_t.plot([], [], 'b-', alpha=0.6, label='Surf T')
    ax_t.set_ylabel('Temp (C)'); ax_t.legend(loc=2); ax_t.grid(alpha=0.2)
    ax_t.set_ylim(20, 50)
    
    ln_sev, = ax_risk.plot([], [], 'y-', label='Severity')
    ax_risk.axhline(1.0, color='r', ls='--'); ax_risk.set_ylabel('Risk'); ax_risk.grid(alpha=0.2)
    ax_risk.set_ylim(0, 5)
    
    ln_soc, = ax_soc.plot([], [], 'c-', lw=2); ax_soc.set_title('SOC'); ax_soc.set_ylim(0,1)
    ln_soh, = ax_soh.plot([], [], 'w-', lw=2); ax_soh.set_title('SOH'); ax_soh.set_ylim(80,105)
    
    # History Buffers
    hist = {k: deque(maxlen=500) for k in ['t', 'pm', 'ps', 'meas', 'tc', 'ts', 'sev', 'soc', 'soh']}
    
    def update(frame):
        while not streamer.queue.empty():
            sample = streamer.queue.get()
            res = engine.step(sample)
            
            hist['t'].append(res['t'])
            hist['pm'].append(res['P_mech'])
            hist['ps'].append(res['P_soft'])
            hist['meas'].append(res['P_meas'])
            hist['tc'].append(res['T_core_est'])
            hist['ts'].append(res['T_surf'])
            hist['sev'].append(res['Severity'])
            hist['soc'].append(res['SOC'])
            hist['soh'].append(res['SOH'])
            
            # Status Text
            ax_info.clear(); ax_info.axis('off')
            c = '#00ff00' if res['Risk']=='normal' else '#ff0000'
            txt = (f"STATUS: {res['Risk'].upper()}\n"
                   f"Sev: {res['Severity']:.2f}\n"
                   f"P_Resid: {res['Resid_Phys']:.1f}\n"
                   f"SOC: {res['SOC']*100:.1f}%\n"
                   f"SOH: {res['SOH']:.1f}%")
            ax_info.text(0.1, 0.5, txt, fontsize=14, color=c, fontfamily='monospace')

        if len(hist['t']) > 1:
            t = list(hist['t'])
            ln_p_mech.set_data(t, list(hist['pm']))
            ln_p_soft.set_data(t, list(hist['ps']))
            ln_p_meas.set_data(t, list(hist['meas']))
            ln_t_core.set_data(t, list(hist['tc']))
            ln_t_surf.set_data(t, list(hist['ts']))
            ln_sev.set_data(t, list(hist['sev']))
            ln_soc.set_data(t, list(hist['soc']))
            ln_soh.set_data(t, list(hist['soh']))
            
            for ax in [ax_p, ax_t, ax_risk, ax_soc, ax_soh]:
                ax.set_xlim(t[0], t[-1]+1)
                
        return ln_p_mech,
    
    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()