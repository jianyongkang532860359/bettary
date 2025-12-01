
-----

````markdown
# Lithium Battery State Estimation & Diagnostics Framework

> **A Physics-Data Hybrid Twin Approach (V13.0 Ultimate Refactoring)**
>
> 基于“机理+数据”融合驱动的锂电池数字孪生、故障诊断与 SOX 估算系统

## 1. 项目概述 (Overview)

本项目构建了一套**车规级**的锂电池全生命周期健康监测算法框架。针对电池**“内部状态不可测”**与**“老化状态难估计”**的行业痛点，本系统采用了 **物理模型 (Physics-based)** 与 **数据驱动 (Data-driven)** 深度融合的策略。

在仅依赖外部易测信号（$V, I, T_{surf}, \epsilon_{case}$）的前提下，系统实现了从**毫秒级故障诊断**到**长周期寿命预测**的全栈能力。

**核心能力：**

* **多维状态反演**：基于 **混合训练策略 (Hybrid Training)** 的软传感器，解决了 AI 模型在动态工况下的泛化问题，精准反演**内部气压 ($P_{gas}$)** 与 **核心温度 ($T_{core}$)**。
* **物理/传感故障解耦**：利用物理残差 ($P_{mech}-P_{soft}$) 与测量残差 ($P_{meas}-P_{soft}$) 的正交特性，有效区分 **真实物理故障 (Cold Swelling)** 与 **传感器漂移 (Sensor Drift)**。
* **高鲁棒 SOX 估算**：内置安时积分 SOC 与 **基于 RLS 的自适应 SOH 估算**，在物理闭环仿真中实现了容量从 100% 向 90% 真值的精准收敛。
* **量化安全评估**：引入 **归一化严重度 (Normalized Severity)** 指标与 **鲁棒阈值标定 (Median+MAD)**，提供红/橙/蓝分级预警，消除虚警。

-----

## 2. 技术架构 (System Architecture)

系统采用模块化分层设计，核心算法位于 `src/` 目录，主流程由 `main_pipeline.py` 驱动：

| 层级 | 模块文件 | 功能定义 | 核心技术 |
| :--- | :--- | :--- | :--- |
| **D0** | `data_schema.py` | **数据底座** | 统一数据语义，严格隔离物理真值 (True)、测量值 (Meas) 与估计值 (Est)。 |
| **D1** | `thermal_model_rc.py` | **热观测器** | 基于 3-Node RC 热网络，提供热力学基础状态。 |
| **D2** | `mech_strain_pressure.py` | **力学观测器** | 建立壳体应变-压力的非线性本构方程，作为物理一致性基准。 |
| **D3** | `soft_sensor.py` | **AI 软测量** | **[V13升级]** 采用“标定+车队(正常)”混合训练，并引入去偏 (De-biasing) 与平滑逻辑，确保 $T_{core}$ 预测的物理真实性。 |
| **D4** | `diagnostics.py` | **诊断核心** | **[V13升级]** 基于 Median+MAD 的鲁棒统计标定，计算归一化严重度，实现红(物理)/橙(传感)/蓝(正常) 分级诊断。 |
| **SOX**| `main_pipeline.py` | **状态估算** | **[V13升级]** 内置 SOC/SOH 估算器。SOH 采用物理闭环验证，展示了容量衰减的实时跟踪能力。 |
| **Data** | `synthetic_data.py` | **数字孪生** | 生成包含正常、冷鼓胀、传感器漂移及**加速老化**的全场景合成数据。 |

-----

## 3. 快速开始 (Quick Start)

### 3.1 环境准备

```bash
pip install -r requirements.txt
````

### 3.2 运行全链路仿真 (V13.0 Pipeline)

该脚本将执行数据生成、模型训练 (D2/D3)、SOX 估算、诊断逻辑 (D4) 及论文绘图：

```bash
python main_pipeline.py
```

-----

## 4\. 关键结果解读 (Key Results)

系统运行后将在 `outputs/` 目录生成一系列**论文级插图**：

### 4.1 故障解耦诊断 (`plot_fault_*.png`)

  * **冷鼓胀 (Cold Swelling)**:
      * 物理模型 (Mech Est) 紧跟故障阶跃，而软传感器 (Soft Est) 保持平稳。
      * **结论**：物理一致性残差显著，判定为电池本体结构故障。
  * **传感器漂移 (Sensor Drift)**:
      * 测量值 (Meas P) 独自漂移，而 Mech Est 与 Soft Est 保持一致且稳定。
      * **结论**：测量残差显著，判定为传感器故障，避免误报。

### 4.2 寿命状态估算 (`plot_sox_estimation.png`)

  * **SOC**: 展示了容量衰减导致的 SOC 估算偏差。
  * **SOH Convergence**: 蓝线 (Est SOH) 从初始 100% 快速、平滑地**收敛至真值 90%**。
      * 验证了算法对 $dQ/dSOC$ 变化的敏感性及物理约束的有效性。

### 4.3 车队健康概览 (`risk_score_*.png`)

  * **Fleet View (`risk_score_fleet.png`)**:
      * 仅展示运营车辆。正常车 (Severity $\approx 0$) 与故障车 (Severity $\gg 1$) 区分度极高。
  * **Validation View (`risk_score_full.png`)**:
      * 展示全量数据（含标定与老化测试），用于验证算法在边缘工况下的鲁棒性。

### 4.4 诊断原理图 (`plot_metrics_scatter.png`)

  * **正交分离**：展示了物理故障和传感器故障在二维残差平面上的清晰聚类，完美位于第一象限的不同区域。

-----

## 5\. 版本演进 (Version History)

### [v13.0] - Ultimate Refactoring (Current)

  * **SOH Physics Loop**: 重构了老化数据的物理闭环生成逻辑，确保 SOH 算法在数学上可观测、可收敛。
  * **Robust Soft Sensor**: 引入混合训练集 (Hybrid Training) 与去偏平滑后处理，彻底解决了 AI 模型在动态工况下对核心温度 ($T_{core}$) 预测失真的问题。
  * **Statistical Thresholds**: 采用 Median+MAD 替代 Mean+Std 进行阈值标定，提升了对离群点的鲁棒性。

### [v9.0 - v11.0] - Engineering Refactoring

  * **Severity Metric**: 引入归一化严重度指标，替代了不稳定的 Isolation Forest 分数。
  * **Auto-Calibration**: 实现了基于 $3\sigma$ / $6\sigma$ 的阈值自动标定算法。

### [v5.0] - Baseline Release

  * 建立了 D1-D4 的基础物理-数据融合框架，实现了初步的压力故障解耦。

-----

**© 2025 Battery Digital Twin Project**

```
```
