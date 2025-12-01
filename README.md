

````markdown
# Lithium Battery State Estimation & Diagnostics Framework

> **A Physics-Data Hybrid Twin Approach (V14.3 Real-Time HIL Ready)**
>
> 基于“机理+数据”融合驱动的锂电池数字孪生、故障诊断与 SOX 估算系统。
> **本项目采用离线/在线分离架构，已做好 MCU 移植准备 (MCU-Ready)。**

## 1. 项目概述 (Overview)

本项目构建了一套**车规级**的锂电池全生命周期健康监测算法框架。针对电池**“内部状态不可测”**与**“老化状态难估计”**的行业痛点，本系统采用了 **物理模型 (Physics-based)** 与 **数据驱动 (Data-driven)** 深度融合的策略。

在仅依赖外部易测信号（$V, I, T_{surf}, \epsilon_{case}$）的前提下，系统实现了从**毫秒级故障诊断**到**长周期寿命预测**的全栈能力。

**核心能力：**

* **多维状态反演**：基于 **混合训练策略** 的软传感器，精准反演**内部气压 ($P_{gas}$)** 与 **核心温度 ($T_{core}$)**。
* **物理/传感故障解耦**：利用物理残差 ($P_{mech}-P_{soft}$) 与测量残差 ($P_{meas}-P_{soft}$) 的正交特性，有效区分 **真实物理故障 (Cold Swelling)** 与 **传感器漂移 (Sensor Drift)**。
* **高鲁棒 SOX 估算**：内置安时积分 SOC 与 **基于 RLS 的自适应 SOH 估算**，在物理闭环仿真中实现了容量从 100% 向 90% 真值的精准收敛。
* **量化安全评估**：引入 **归一化严重度 (Normalized Severity)** 指标与 **鲁棒阈值标定**，提供红/橙/蓝分级预警。
* **实时 HIL 仿真**：提供独立的 `OnlineBMSEngine`，支持逐点流式计算，模拟真实 BMS 固件运行环境。

-----

## 2. 代码结构与功能说明 (File Structure & Functions)

本项目分为 **主程序 (Applications)** 与 **核心算法库 (Core Library / src)** 两大部分。

### 2.1 主程序 (Applications)

| 文件名 | 功能定义 | 核心作用 |
| :--- | :--- | :--- |
| **`main_pipeline.py`** | **离线全链路仿真主程序** | 系统的“大脑”与“工厂”。负责生成合成数据、训练 D2/D3 模型、执行 SOH 物理闭环验证、自动标定诊断阈值，并生成所有论文级分析图表 (`outputs/`)。 |
| **`realtime_monitor.py`** | **实时 HIL 仿真上位机** | 系统的“仪表盘”。模拟传感器数据流，实时调用在线引擎，动态展示压力、温度、SOC/SOH 曲线及故障严重度报警。 |
| `analyze_metrics.py` | 离线指标分析工具 | (可选) 专门用于批量计算和导出诊断指标统计表 (`diagnosis_indicators_table.csv`)。 |
| `plot_2d_scatter.py` | 散点图绘制工具 | (可选) 专门用于绘制高精度的故障解耦二维散点图。 |

### 2.2 核心算法库 (src/)

| 模块文件 | 角色 | 功能详解 |
| :--- | :--- | :--- |
| **`online_engine.py`** | **RT 在线引擎** | **[核心交付物]** BMS 实时推理内核。封装了 SOC/SOH 估算、软传感器推理及诊断逻辑。**特点：** 去除了 Pandas 依赖，支持 `step(sample)` 逐点计算，代码结构可直接对标嵌入式 C 代码。 |
| **`mech_strain_pressure.py`** | **D2 力学模型** | **物理观测器**。定义了壳体应变 ($\epsilon$) 与内部气压 ($P$) 的非线性本构方程，包含热膨胀补偿逻辑。提供 `fit` (参数辨识) 和 `predict` 接口。 |
| **`soft_sensor.py`** | **D3 软传感器** | **AI 观测器**。基于 XGBoost 的数据驱动模型。包含特征工程（滑动窗口、微分）逻辑，负责从 $V, I, T$ 反演内部 $P$ 和 $T_{core}$。 |
| **`thermal_model_rc.py`** | **D1 热模型** | **热力学基准**。基于 3-Node RC 网络的热模型，用于生成仿真数据中的核心温度真值，提供物理参考。 |
| **`diagnostics.py`** | **D4 诊断工具** | **特征提取库**。负责计算各类残差指标（一致性残差、测量残差）及特征聚合，为上层诊断逻辑提供数据基础。 |
| **`synthetic_data.py`** | **数据工厂** | **数字孪生生成器**。生成包含正常循环、冷鼓胀 (Overpressure)、传感器漂移 (Sensor Drift) 及加速老化 (Aging) 的高保真合成数据。 |
| `data_loader.py` | 数据适配器 | 负责读取外部 CSV/Excel 实验数据，进行列名映射、单位换算及时间轴对齐。 |
| `preprocessing.py` | 信号预处理 | 提供低通滤波 (Low-pass Filter) 和去噪功能，提升输入数据质量。 |
| `data_schema.py` | 数据字典 | 定义系统的标准字段命名（如 `I_A`, `V_V`）与物理单位，确保全链路数据语义一致。 |

-----

## 3. 快速开始 (Quick Start)

### 3.1 环境准备

```bash
pip install -r requirements.txt
````

### 3.2 模式一：离线全链路分析 (Offline Pipeline)

用于算法验证、参数标定及生成分析报告。

```bash
python main_pipeline.py
```

**输出**：`outputs/` 目录下生成故障解耦图、SOH 收敛曲线及车队健康报表。

### 3.3 模式二：实时 HIL 仿真 (Real-time Dashboard)

模拟真实 BMS 运行环境，体验流式数据处理与动态报警。

```bash
python realtime_monitor.py
```

**操作**：运行后将弹出实时动态窗口，展示压力、温度、SOC/SOH 及故障严重度的实时变化曲线。

-----

## 4\. 关键结果解读 (Key Results)

### 4.1 故障解耦诊断 (`outputs/plot_fault_*.png`)

  * **冷鼓胀 (Cold Swelling)**: 物理模型 (Mech) 紧跟故障阶跃，软传感器 (Soft) 保持平稳，物理残差显著。
  * **传感器漂移 (Sensor Drift)**: 测量值 (Meas) 独自漂移，模型保持一致，测量残差显著。

### 4.2 寿命状态估算 (`outputs/plot_sox_estimation.png`)

  * **SOH Convergence**: 蓝线 (Est SOH) 从初始 100% 快速、平滑地**收敛至真值 90%**，验证了算法对 $dQ/dSOC$ 变化的敏感性。

### 4.3 实时监控看板 (`realtime_monitor.py` 界面)

  * **Severity 仪表**: 实时显示归一化严重度。正常工况下 $<0.5$，故障发生时迅速飙升至 $>1.0$ 并触发红色警报。
  * **动态响应**: 可直观观测到 SOH 在老化阶段的动态调整过程。

-----

## 5\. 版本演进 (Version History)

### [v14.3] - Real-Time HIL Ready (Current)

  * **Online Engine**: 封装了 `OnlineBMSEngine` 类，实现了算法的流式化重构。
  * **HIL Dashboard**: 新增 `realtime_monitor.py`，提供实时数据回放与可视化监控。

### [v13.0] - Ultimate Refactoring

  * **SOH Physics Loop**: 重构了老化数据的物理闭环生成逻辑，确保 SOH 算法在数学上可观测。
  * **Robust Soft Sensor**: 引入混合训练与去偏平滑，解决了 AI 温度失真问题。

### [v9.0 - v11.0] - Engineering Refactoring

  * **Severity Metric**: 引入归一化严重度指标与鲁棒阈值标定。

-----

**© 2025 Battery Digital Twin Project**

```
```