# 代码审查：锂电池数字孪生与故障诊断系统

## 1. 嵌入式移植可行性

### [Critical] 运行时仍依赖 sklearn/xgboost 管线
- 发现问题：在线引擎在 `__init__` 中直接持有 `StandardScaler` + `MultiOutputRegressor(XGBRegressor)` 对象用于推理。 【F:src/online_engine.py†L64-L151】
- 解释后果：在 MCU/FPGA 上无法部署 sklearn/xgboost Python 对象；即便通过 ctypes，内存占用和线程安全也无法保证，移植将被卡住。
- 修改代码：训练阶段导出纯 Booster & 归一化参数，在线侧用纯 Numpy 做手工标准化+树推理接口（treelite/xgboost C predictor）。
  ```python
  # 训练后导出
  booster = est.get_booster()
  np.savez('soft_norm.npz', mean=self.scaler.mean_, scale=self.scaler.scale_)

  # 在线侧加载
  norm = np.load('soft_norm.npz')
  feats = (feats_arr - norm['mean']) / norm['scale']
  p_soft, t_core = booster.inplace_predict(feats)[0]
  ```

### 滚动特征构造与离线不一致
- 发现问题：离线特征工程包含 `diff()`、多窗口 `mean/std`，并保留 `dT_surf_amb` 等字段；在线版本手写特征顺序与数量不同且未对齐窗口 30/60，遇到维度不符直接返回默认值。 【F:src/online_engine.py†L104-L151】【F:src/soft_sensor.py†L73-L151】
- 解释后果：上线后软传感器可能永远输出 0 或退化为表面温度，导致诊断残差失真，物理/传感器故障无法区分。
- 修改代码：把 `_feat_eng_group_logic` 逻辑提炼为纯函数供在线侧共享；在线侧维护固定长度环形缓冲+运行和/方差累积，严格按照同一列顺序拼装特征向量。
  ```python
  # soft_sensor.py 中提炼
  def build_features_from_buffer(buf, config: ModelConfig) -> np.ndarray:
      # 使用运行和/平方和避免逐次 np.mean/np.std
      ...

  # online_engine.py 中调用
  feats = build_features_from_buffer(self.buf_stats, self.config)
  ```

### 历史缓存使用 deque + list 拷贝
- 发现问题：`_get_soft_estimates` 每步将 deque 转成 list，再对每个窗口遍历求均值/方差。 【F:src/online_engine.py†L131-L141】
- 解释后果：O(N) 拷贝在 50 Hz 刷新下容易占满 SRAM，且 NumPy mean/std 需要动态堆分配；MCU 上不可接受。
- 修改代码：维护每个特征的运行和/平方和与计数，使用定长数组实现环形缓冲，避免对象拷贝。
  ```python
  # 初始化
  self.buf = np.zeros((self.maxlen, n_feat))
  self.idx = 0
  self.count = 0
  # 更新
  self.buf[self.idx] = curr_vals
  self.idx = (self.idx + 1) % self.maxlen
  self.count = min(self.count + 1, self.maxlen)
  # 运行均值/方差
  mean_w = run_sum[w] / w
  var_w = (run_sq[w] / w) - mean_w**2
  ```

### SOH 更新依赖真实 SOC，未做电流窗口过滤
- 发现问题：`OnlineSOHEstimator` 直接使用 `sample['SOC']` 作为真值增量，且只基于累积 dSOC 阈值触发，不校验当前是否处于静置或低流段。 【F:src/online_engine.py†L167-L176】
- 解释后果：现场缺少真值 SOC 时 `d_soc_true` 恒为 0，SOH 永不更新；在低电流噪声下仍会累计 `acc_dq` 造成容量估计偏大。
- 修改代码：改为基于估计 SOC/OCV 校正的 ΔSOC；增加电流幅值和最小持续时间门限，并在静置段冻结积分。
  ```python
  if abs(i_a) < I_MIN or dt_acc < T_MIN:
      return self.soh
  d_soc_est = self.soc_observer.estimate_delta(...)
  ```

## 2. 算法逻辑与数值稳定性

### 严重度判定无去抖/滞回
- 发现问题：`Severity` 直接由单点残差计算，没有低通或计数器，阈值比较立即跳转风险状态。 【F:src/online_engine.py†L182-L191】
- 解释后果：在噪声或边界情况下状态会在 `normal`/`fault` 之间抖动，影响告警可信度。
- 修改代码：增加双阈值滞回和计数去抖，例如连续 N 次超阈才置故障，连续 M 次低于下阈才恢复。
  ```python
  if norm_rc > HIGH_TH or norm_rm > HIGH_TH: fault_cnt += 1
  if fault_cnt >= N_FAULT: risk = ...
  elif norm_rc < LOW_TH and norm_rm < LOW_TH: recover_cnt += 1
  ```

### 滚动窗口冷启动处理缺失
- 发现问题：软传感器在 `len(history) < 1` 时返回 0 或表面温度，占据诊断残差；未做逐步升温/掩码。 【F:src/online_engine.py†L100-L152】
- 解释后果：前 N 秒内输出假零值导致残差虚高，引发误报。
- 修改代码：增加 warmup 计数与输出掩码，在窗口填满前不计算风险或仅输出 `None` 并跳过诊断。
  ```python
  if self.count < MIN_WARMUP:
      return {'P_soft': None, 'Risk': 'warming_up', ...}
  ```

## 3. 代码质量与工程规范

### DataStreamer 缺少停止/背压，队列可能无限增长
- 发现问题：实时监控线程无 `stop()`，队列容量无限，消费者阻塞时会持续累积。 【F:realtime_monitor.py†L24-L170】
- 解释后果：长时间运行可能导致主机或 HMI 内存飙升；也无法在退出时优雅关闭线程。
- 修改代码：为队列设置 `maxsize` 并实现 `stop()`，循环时使用 `queue.put(sample, timeout=...)` 捕获阻塞，退出时调用 `join()`。
  ```python
  self.queue = queue.Queue(maxsize=200)
  def stop(self):
      self.running = False
  def _worker(...):
      self.queue.put(sample, timeout=0.1)
  ```

### 类型提示与输入校验缺失
- 发现问题：`OnlineBMSEngine.step` 等函数未声明输入类型/范围，`sample` 中缺少键时默认为 0 或 NaN，异常 silent ignore。 【F:src/online_engine.py†L153-L199】
- 解释后果：集成时很难发现上游信号缺失，导致算法输出被 0 污染。
- 修改代码：为 `step`/`update` 补充类型注解、必需字段校验与断言，或在嵌入式侧返回错误码。
  ```python
  def step(self, sample: Mapping[str, float]) -> dict:
      required = ('I_A','eps_eq','T_surf_degC')
      for k in required:
          if k not in sample:
              raise ValueError(f"missing {k}")
  ```
