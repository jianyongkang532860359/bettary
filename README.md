# Lithium Battery State Estimation & Diagnostics Framework

> **A Physics-Data Hybrid Twin Approach**
>
> 基于“机理+数据”融合驱动的锂电池数字孪生与预警系统

## 1. 项目概述 (Overview)

本项目构建了一套完整的、逻辑自洽的锂电池健康监测算法框架。针对电池**“内部状态（核心温度、气压）不可直接测量”**的行业痛点，本系统采用 **物理模型 (Physics-based)** 与 **数据驱动 (Data-driven)** 相结合的“双轮驱动”策略。

在仅依赖外部易测信号（电压、电流、表面温度、壳体应变）的前提下，系统实现了对电池内部状态的**毫秒级软测量**与**故障解耦诊断**。

**核心能力：**

* **高精度软测量**：基于 XGBoost 的非线性回归，反演精度 $R^2 > 0.99$ (MAE $\approx$ 0.1 kPa)。
* **故障解耦**：有效区分 **真实物理故障 (Overpressure)** 与 **传感器漂移 (Sensor Fault)**，解决了传统BMS“误报率高”的难题。
* **分级预警**：基于 Isolation Forest 的无监督异常评分，无需故障样本训练即可识别异常。

-----

## 2. 技术架构 (System Architecture)

系统采用模块化分层设计，核心代码位于 `src/` 目录下：

| 层级 | 模块文件 | 功能定义 | 核心技术 |
| :--- | :--- | :--- | :--- |
| **D0** | `data_schema.py` | **数据底座** | 统一数据语义，严格隔离物理真值 (True)、测量值 (Meas) 与估计值 (Est)。 |
| **D1** | `thermal_model_rc.py` | **热观测器** | 基于 3-Node RC 热网络 + SciPy 加速，实时反演核心温度 ($T_{core}$)。 |
| **D2** | `mech_strain_pressure.py` | **力学观测器** | 壳体应变-压力非线性映射（含热补偿），支持从标定数据辨识参数。 |
| **D3** | `soft_sensor.py` | **AI 软测量** | 基于 XGBoost 的数据驱动模型，学习物理规律，反演内部气压。 |
| **D4** | `diagnostics.py` | **诊断大脑** | 计算物理一致性残差 ($P_{mech} - P_{soft}$) 与测量残差，利用 Isolation Forest 异常检测。 |
| **Data** | `synthetic_data.py` | **数据工厂** | 生成包含正常、过压、传感器故障的多场景合成数据，注入真实噪声。 |
| **Main** | `main_pipeline.py` | **全链路集成** | 一键运行数据生成、模型训练、推理评估与可视化报告。 |

-----

## 3. 快速开始 (Quick Start)

### 3.1 环境准备

确保已安装 Python 3.9+，建议使用 Conda 环境：

```bash
conda create -n battery_env python=3.9
conda activate battery_env
pip install -r requirements.txt
