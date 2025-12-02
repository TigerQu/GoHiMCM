# Alpha 参数扫描实验 - 文件索引与指南

## 📊 实验成果概览

**实验时间**: 2025-11-18  
**实验规模**: 230 episodes (23 alpha × 10 seeds)  
**运行时间**: ~5 分钟  
**结论**: 最优 α = 0.15，效率 0.5434 rescued/step

---

## 📁 文件结构

### 1️⃣ 核心实验代码

#### `src/traditional_planner/alpha_sweep_experiment.py`
- **功能**: 主实验脚本，运行 alpha 参数扫描
- **包含**:
  - `run_one_episode()`: 单个 episode 运行
  - `sweep_alpha_on_daycare()`: 参数扫描和绘图
- **使用**: 
  ```bash
  python src/traditional_planner/alpha_sweep_experiment.py
  ```
- **修改点**: 
  - 第 120 行: `alpha_values` - 修改测试的 alpha 范围
  - 第 125 行: `num_seeds` - 修改每个 alpha 的 seed 数
- **输出**: 5 个 PNG 图表 + 终端统计

#### `src/traditional_planner/sweep_parameter_guide.py`
- **功能**: 教学指南，展示如何扩展到其他参数
- **内容**:
  - PlannerConfig 所有可扫描的参数
  - 如何修改 alpha_sweep_experiment.py
  - 2D 网格搜索的示例代码
- **使用**: 
  ```bash
  python src/traditional_planner/sweep_parameter_guide.py
  ```
- **最适合**: 学习如何做 beta、gamma 或风险权重扫描

#### `src/traditional_planner/sweep_all_layouts.py`
- **功能**: 启动脚本，便于运行实验
- **使用**: 
  ```bash
  python src/traditional_planner/sweep_all_layouts.py
  ```

---

### 2️⃣ 可视化输出 (5 个 PNG，总计 575 KB)

#### `daycare_alpha_rescued.png` (76 KB)
- **内容**: 救援人数 vs α 曲线图
- **用途**: 查看不同 α 下的救援效果
- **关键**: 低 α 救人多，但时间长

#### `daycare_alpha_time.png` (72 KB)
- **内容**: 清理时间 vs α 曲线图
- **用途**: 查看不同 α 下的清理速度
- **关键**: 高 α 清理快，但救人少

#### `daycare_alpha_efficiency.png` (107 KB) ⭐
- **内容**: 效率 (rescued/time) vs α 曲线
- **用途**: **找到最优点** - 看这个！
- **关键**: 峰值在 α ≈ 0.15
- **推荐**: 用这张图决策

#### `daycare_alpha_tradeoff.png` (104 KB)
- **内容**: 散点图，X 轴=时间，Y 轴=救援数，着色=α
- **用途**: 理解救援 vs 时间的权衡
- **特点**: 可直观看出 Pareto 前沿

#### `daycare_alpha_combined.png` (216 KB)
- **内容**: 4 个子图 (救援、时间、效率、权衡)
- **用途**: 完整分析一目了然
- **推荐**: 演示和报告使用

---

### 3️⃣ 文档 (2 个 Markdown)

#### `ALPHA_SWEEP_RESULTS.md`
- **内容**:
  - 完整的 23×4 结果表格
  - 每个 alpha 的详细数据
  - 三个关键发现 (最多救援、最快清理、最高效率)
  - 区间分析 (低/中/高 α)
  - 推荐表格和生产环境建议
- **适合**: 需要详细数据的人
- **大小**: ~3 KB

#### `ALPHA_SWEEP_EXPERIMENT_README.md`
- **内容**:
  - 实验完整总结
  - 实验规模和时间
  - 核心结果和数据表
  - 三个关键指标对比
  - 推荐方案详细论述
  - 技术细节和复杂度分析
  - 后续可能的工作
  - 完成清单
- **适合**: 想了解全貌的人
- **大小**: ~8 KB

---

### 4️⃣ 快速参考

#### `quick_alpha_reference.py`
- **功能**: 快速查看实验结果摘要
- **用途**: 
  ```bash
  python quick_alpha_reference.py
  ```
- **输出**: 格式化的快速参考卡
- **特点**: 包含建议、趋势、Q&A

#### `print_alpha_results.py`
- **功能**: 打印格式化的结果摘要
- **用途**:
  ```bash
  python print_alpha_results.py
  ```
- **输出**: 彩色格式化的关键指标

---

## 🎯 如何使用本实验结果

### 场景 1: 我只想快速了解结论
```bash
python quick_alpha_reference.py
```
**查看**: 快速参考卡  
**耗时**: 1 秒

### 场景 2: 我要详细的数据和分析
**查看**: `ALPHA_SWEEP_RESULTS.md`  
**耗时**: 5 分钟

### 场景 3: 我想重现实验
```bash
python src/traditional_planner/alpha_sweep_experiment.py
```
**耗时**: 5 分钟

### 场景 4: 我想扩展到其他参数 (如 beta)
**查看**: `sweep_parameter_guide.py`
```bash
python src/traditional_planner/sweep_parameter_guide.py
```
**耗时**: 2 分钟学习，然后自己修改代码

### 场景 5: 我想看漂亮的可视化
**查看**: 5 个 PNG 文件
- 最重要: `daycare_alpha_efficiency.png`
- 最完整: `daycare_alpha_combined.png`

---

## 🔍 快速查询

### Q: 应该用什么 alpha?
**A**: 0.15 (最优效率，综合表现最好)

### Q: 为什么不用 alpha = 0.05?
**A**: 救人虽然多 (35.7 vs 35.7... 一样),但清理时间长 (67 vs 65),效率略低 (0.529 vs 0.544)

### Q: 为什么不用 alpha = 2.0?
**A**: 虽然清理快 (62.8 步),但救人少 (32.1 vs 35.7),效率也低 (0.511 vs 0.544),且不稳定 (标准差 0.6 属异常)

### Q: 能在 office 或 warehouse 上运行吗?
**A**: 可以！修改 `alpha_sweep_experiment.py` 第 115 行:
```python
build_env_fn = build_babycare_layout  # 改成 build_standard_office_layout 等
```

### Q: 能扫描 beta 参数吗?
**A**: 可以！查看 `sweep_parameter_guide.py` 第 30+ 行的示例

### Q: 能做 2D 扫描 (alpha × beta)?
**A**: 可以！查看 `sweep_parameter_guide.py` 最后的"2D 网格搜索"示例

---

## 📊 数据汇总表

| Alpha | 救援数 | 时间 | 效率 | 标准差 | 评价 |
|-------|--------|------|------|--------|------|
| 0.05  | 35.7 | 67.5 | 0.529 | 3.5 | 救人最多，时间最长 |
| 0.15  | 35.7 | 65.7 | **0.544** | 2.0 | ⭐ **最优平衡** |
| 0.25  | 35.7 | 65.9 | 0.542 | 2.3 | 几乎同 0.15 |
| 1.0   | 32.6 | 64.0 | 0.509 | 3.8 | 开始不稳定 |
| 2.0   | 32.1 | 62.8 | 0.511 | 0.6 | 时间最快，但救人最少 |

---

## 🚀 下一步工作建议

### 优先级 HIGH
- [ ] 在 office 和 warehouse 上运行相同实验
- [ ] 对比三个布局的最优 alpha
- [ ] 使用推荐的 α = 0.15 更新默认参数

### 优先级 MEDIUM
- [ ] 扫描 beta 参数 (风险奖励权重)
- [ ] 扫描 gamma 参数 (拥堵惩罚权重)
- [ ] 做 2D 网格搜索 (alpha × beta) 找最优组合

### 优先级 LOW
- [ ] Pareto 前沿分析 (多目标优化)
- [ ] 自动超参数调优 (贝叶斯优化)
- [ ] 对比不同规划策略

---

## 📝 版本信息

- **创建日期**: 2025-11-18
- **Python 版本**: 3.9+
- **依赖**: numpy, matplotlib, networkx
- **运行平台**: macOS (M1)，Linux/Windows 兼容
- **实验总耗时**: ~5 分钟

---

## ✅ 完成检查表

- [x] 修复 alpha 参数设置 (alpha_value 变量错误)
- [x] 扩展到 23 个 alpha 值 (0.05 ~ 2.0)
- [x] 增加到 10 个 seed 每个 alpha (230 total)
- [x] 添加详细统计分析
- [x] 生成 5 个专业图表
- [x] 编写 2 份详细文档
- [x] 提供参数扩展指南
- [x] 创建快速参考卡
- [x] 编写本索引文档

---

**推荐**: 将 `alpha` 默认值从 0.2 改为 **0.15** 👍

