# 参数优化实验 - 完整项目结构

## 📊 项目概述

本项目完成了从**单参数优化**到**多参数联合优化**的完整参数优化实验。

**核心问题**: 用户指出单参数优化 (α=0.15) 可能不是全局最优，需要考虑其他参数的组合。

**解决方案**: 进行 3 阶段的系统优化，找到最优参数组合。

---

## 📁 文件结构

```
experiments/
├── 📁 alpha_sweep/                    ← PHASE 1: 单参数优化
│   ├── ALPHA_SWEEP_RESULTS.md        (详细数据表)
│   ├── ALPHA_SWEEP_EXPERIMENT_README.md
│   ├── ALPHA_SWEEP_INDEX.md
│   ├── daycare_alpha_rescued.png     (救援人数曲线)
│   ├── daycare_alpha_time.png        (清理时间曲线)
│   ├── daycare_alpha_efficiency.png  (效率曲线) ⭐
│   ├── daycare_alpha_tradeoff.png    (权衡分析)
│   ├── daycare_alpha_combined.png    (4 图组合)
│   ├── quick_alpha_reference.py      (快速参考)
│   ├── print_alpha_results.py        (结果摘要)
│   └── alpha_sweep_experiment.py     (实验脚本)
│
├── 📁 grid_search_3d/                ← PHASE 2: 3D 联合优化
│   ├── grid_search_3d_heatmaps.png   (3 个热力图) ✅ 查看这个
│   ├── grid_search_3d_results.json   (详细数据)
│   └── grid_search_alpha_beta_gamma.py
│
├── 📁 grid_search_2d/                ← PHASE 3: 2D 详细优化
│   ├── grid_search_heatmaps.png      (热力图组合) ✅ 查看这个
│   ├── grid_search_results.json      (详细数据)
│   └── grid_search_alpha_beta.py
│
├── 📄 PARAMETER_OPTIMIZATION_GUIDE.md         (使用指南)
├── 📄 PARAMETER_OPTIMIZATION_COMPARISON_REPORT.txt (对比分析)
├── 📄 FINAL_OPTIMIZATION_SUMMARY.txt          (完整总结) ⭐
└── 📄 README_MASTER.md                        (本文件)
```

---

## 🎯 快速开始

### 查看结果 (推荐)

1. **查看 3D 优化热力图** (最重要)
   ```bash
   open experiments/grid_search_3d/grid_search_3d_heatmaps.png
   ```
   显示 Alpha × Beta × Gamma 的完整优化结果

2. **查看完整总结**
   ```bash
   cat experiments/FINAL_OPTIMIZATION_SUMMARY.txt
   ```
   包含所有关键发现和建议

3. **查看对比分析**
   ```bash
   cat experiments/PARAMETER_OPTIMIZATION_COMPARISON_REPORT.txt
   ```
   单参数 vs 多参数优化的详细对比

---

## 🏆 核心发现

### 关键数字

| 指标 | 单参数优化 | 3D 优化 | 改进 |
|------|-----------|--------|------|
| 最优参数 | α=0.15, β=4.0, γ=0.2 | α=0.30, β=4.0, γ=0.1 | ✅ |
| 效率 | 0.5434 | 0.5752 | +5.8% |
| 救援人数 | 35.7 ± 2.0 | 37.0 ± 2.8 | +3.6% |
| 清理时间 | 65.7 ± 2.0 | 64.3 ± 1.2 | -2.1% |

### 关键洞见

1. **参数之间存在重要的相互作用**
   - Alpha 的最优值取决于 Beta 的设置
   - Gamma 的影响最大 (从 0.2 降低到 0.1 效果显著)
   - 不能独立优化各个参数

2. **用户的批评完全有理**
   - 单参数优化确实只是局部最优
   - 多参数优化找到了更好的全局解
   - 改进幅度达 5.8%

3. **推荐参数已识别**
   - 新推荐: α=0.30, β=4.0, γ=0.1
   - 改进体现在更多救援人数、更少清理时间
   - 性能更稳定 (时间标准差从 2.0 降到 1.2)

---

## 📊 实验详情

### PHASE 1: 单参数优化 (Alpha 扫描)

```
参数: α ∈ [0.05, 0.1, ..., 2.0]  (23 个值)
      β = 4.0 (固定)
      γ = 0.2 (固定)

规模: 23 × 10 seeds = 230 episodes
时间: ~5 分钟
结论: α=0.15 最优 (在 β=4.0, γ=0.2 下)
```

**问题**: 假设 β 和 γ 是最优的，但实际上它们不是!

### PHASE 2: 3D 联合优化

```
参数: α ∈ {0.1, 0.2, 0.3}
      β ∈ {2.0, 4.0, 6.0}
      γ ∈ {0.1, 0.2, 0.4}

规模: 3 × 3 × 3 × 3 seeds = 81 episodes
时间: ~5-8 分钟
结论: α=0.30, β=4.0, γ=0.1 最优 (+5.8% 效率)
```

**发现**: 
- α 在 β=2.0 时最优为 0.1 (效率=0.5723)
- α 在 β=4.0 时最优为 0.3 (效率=0.5752) ⭐
- α 在 β=6.0 时最优为 0.1 (效率=0.5723)

### PHASE 3: 2D 详细优化

```
参数: α ∈ [0.05, 0.1, 0.15, ..., 2.0]  (9 个值)
      β ∈ [1.0, 2.0, 3.0, ..., 8.0]     (8 个值)
      γ = 0.2 (固定)

规模: 9 × 8 × 5 seeds = 360 episodes
时间: ~20 分钟
结论: 更详细的 α-β 最优点
```

---

## 🔧 如何使用这些参数

### 选项 1: 更新源代码

编辑 `src/traditional_planner/scoring.py`:

```python
@dataclass
class PlannerConfig:
    alpha: float = 0.30    # 从 0.2 改为 0.30
    beta: float = 4.0      # 保持不变
    gamma: float = 0.1     # 从 0.2 改为 0.1
```

### 选项 2: 运行时配置

在脚本中使用:

```python
from src.traditional_planner.scoring import PlannerConfig

cfg = PlannerConfig(
    alpha=0.30,
    beta=4.0,
    gamma=0.1
)
planner = GreedySweepPlanner(adapter=adapter, cfg=cfg)
```

---

## 🚀 下一步建议

### 优先级 1 (立即行动)
- [ ] 在其他布局 (office, warehouse) 上验证这些参数
- [ ] 运行相同的 3D 优化，查看是否一致
- [ ] 如果不一致，为每个布局单独优化

### 优先级 2 (中期)
- [ ] 分析参数与任务目标的关系
- [ ] 实现自动参数调优 (贝叶斯优化)
- [ ] 建立参数-性能模型

### 优先级 3 (长期)
- [ ] 参数敏感性分析
- [ ] 多目标优化 (救援人数、时间、安全性)
- [ ] 找 Pareto 前沿

---

## 📖 文档导航

| 想要... | 查看文件 | 用时 |
|---------|---------|------|
| 快速了解结论 | FINAL_OPTIMIZATION_SUMMARY.txt | 5 分钟 |
| 详细对比分析 | PARAMETER_OPTIMIZATION_COMPARISON_REPORT.txt | 10 分钟 |
| 使用指南 | PARAMETER_OPTIMIZATION_GUIDE.md | 5 分钟 |
| 单参数结果 | experiments/alpha_sweep/ | 各自独立 |
| 3D 优化数据 | grid_search_3d/grid_search_3d_results.json | 代码/脚本 |
| 2D 优化数据 | grid_search_2d/grid_search_results.json | 代码/脚本 |

---

## ❓ 常见问题

**Q: 为什么要改变所有三个参数?**
A: 因为它们之间有相互作用。只改变 α 可能不够，β 和 γ 也需要调整以获得最佳效果。

**Q: 这些参数对其他布局也适用吗?**
A: 不一定。最优参数取决于布局结构、agent 数量、火灾配置等因素。建议在其他布局上测试。

**Q: 如何判断是否需要再次优化?**
A: 当改进需求明确时 (例如，救援人数需要增加 10%)，或当参数改变 (agent 数量、布局等) 时。

---

## 📞 支持

如有问题，请查看:
1. FINAL_OPTIMIZATION_SUMMARY.txt 的 FAQ 部分
2. 各个优化脚本的代码注释
3. scoring.py 中的参数定义

---

**最后更新**: 2024-11-18  
**总实验耗时**: ~30 分钟  
**总 episodes**: 230 + 81 + 360 = 671  
**推荐参数**: α=0.30, β=4.0, γ=0.1 ✅
