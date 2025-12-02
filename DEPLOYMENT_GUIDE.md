# 🚀 参数部署指南

**用途**: 快速部署最优参数到您的项目中  
**难度**: ⭐ 非常简单  
**耗时**: 5 分钟  

---

## ⚡ 快速部署 (2 分钟)

### 步骤 1: 打开您的代码文件

```python
# 你的代码文件位置，例如:
# src/solutions/trad_v0/runner.py
# 或任何使用 PlannerConfig 的地方
```

### 步骤 2: 导入 PlannerConfig

```python
from src.traditional_planner.scoring import PlannerConfig
```

### 步骤 3: 创建配置对象

**选项 A: 推荐 - 通用参数** ⭐

```python
config = PlannerConfig(
    alpha=0.15,
    beta=6.0,
    gamma=0.1
)
```

**选项 B: 针对 Office 地图**

```python
config = PlannerConfig(
    alpha=0.15,
    beta=4.0,
    gamma=0.1
)
```

**选项 C: 针对 Babycare 地图**

```python
config = PlannerConfig(
    alpha=0.15,
    beta=2.0,
    gamma=0.2
)
```

**选项 D: 针对 Warehouse 地图**

```python
config = PlannerConfig(
    alpha=0.2,
    beta=2.0,
    gamma=0.1
)
```

### 步骤 4: 使用配置

```python
from src.solutions.trad_v0.runner import run_planner

result = run_planner(env, config)
print(f"效率: {result.efficiency}")
print(f"救援人数: {result.rescued}")
print(f"完成时间: {result.time}")
```

### 步骤 5: 运行测试

```bash
cd /Users/hengshao/Desktop/HIMCM/GoHiMCM
python src/solutions/trad_v0/runner.py
```

✅ **完成!**

---

## 📋 部署方案对比

| 方案 | 参数 | 何时选用 | 效率 | 优缺点 |
|------|------|---------|------|--------|
| **推荐** | (0.15, 6.0, 0.1) | ✅ 大多数情况 | 综合1.03 | ✅ 通用, ✅ 可靠 |
| **Office** | (0.15, 4.0, 0.1) | 知道是办公楼 | 0.2619 | ✅ 最优, ❌ 需识别 |
| **Babycare** | (0.15, 2.0, 0.2) | 知道是托儿所 | **0.5779** | ✅ 最优效率, ❌ 需识别 |
| **Warehouse** | (0.2, 2.0, 0.1) | 知道是仓库 | 0.0931 | ✅ 最优, ❌ 需识别 |

---

## 🔧 不同框架的集成方式

### 方式 1: 直接在 runner 中使用

```python
# src/solutions/trad_v0/runner.py

from src.traditional_planner.scoring import PlannerConfig

def run_planner(env, config=None):
    if config is None:
        # 使用推荐的通用参数
        config = PlannerConfig(
            alpha=0.15,
            beta=6.0,
            gamma=0.1
        )
    
    # 继续原有逻辑...
    return planner.run(env, config)
```

### 方式 2: 通过参数传入

```python
def create_config(layout_name=None):
    """根据地图类型创建配置"""
    params = {
        'office': {'alpha': 0.15, 'beta': 4.0, 'gamma': 0.1},
        'babycare': {'alpha': 0.15, 'beta': 2.0, 'gamma': 0.2},
        'warehouse': {'alpha': 0.2, 'beta': 2.0, 'gamma': 0.1},
    }
    
    if layout_name in params:
        p = params[layout_name]
    else:
        # 默认使用推荐参数
        p = {'alpha': 0.15, 'beta': 6.0, 'gamma': 0.1}
    
    return PlannerConfig(**p)


# 使用方式
config = create_config('office')
# 或
config = create_config('warehouse')
# 或使用默认
config = create_config()
```

### 方式 3: 配置文件方式

创建 `config/planner_params.yaml`:

```yaml
# 推荐参数
default:
  alpha: 0.15
  beta: 6.0
  gamma: 0.1

# 各地图最优参数
layouts:
  office:
    alpha: 0.15
    beta: 4.0
    gamma: 0.1
  
  babycare:
    alpha: 0.15
    beta: 2.0
    gamma: 0.2
  
  warehouse:
    alpha: 0.2
    beta: 2.0
    gamma: 0.1
```

然后在代码中:

```python
import yaml
from src.traditional_planner.scoring import PlannerConfig

def load_config(layout_name=None):
    with open('config/planner_params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    if layout_name and layout_name in params['layouts']:
        p = params['layouts'][layout_name]
    else:
        p = params['default']
    
    return PlannerConfig(**p)
```

### 方式 4: 环境变量方式

```bash
# 设置环境变量
export PLANNER_ALPHA=0.15
export PLANNER_BETA=6.0
export PLANNER_GAMMA=0.1
```

```python
import os
from src.traditional_planner.scoring import PlannerConfig

config = PlannerConfig(
    alpha=float(os.getenv('PLANNER_ALPHA', 0.15)),
    beta=float(os.getenv('PLANNER_BETA', 6.0)),
    gamma=float(os.getenv('PLANNER_GAMMA', 0.1))
)
```

---

## ✅ 验证部署

### 方法 1: 直接运行测试

```python
from src.environment.env import EvacuationEnv
from src.traditional_planner.scoring import PlannerConfig
from src.solutions.trad_v0.runner import run_planner

# 创建环境
env = EvacuationEnv(layout_name='office')

# 创建配置
config = PlannerConfig(alpha=0.15, beta=4.0, gamma=0.1)

# 运行
result = run_planner(env, config)

# 验证结果
print(f"✅ 部署成功!")
print(f"   效率: {result.efficiency:.4f}")
print(f"   救援: {result.rescued} 人")
print(f"   耗时: {result.time} 步")
```

### 方法 2: 运行提供的验证脚本

```bash
cd /Users/hengshao/Desktop/HIMCM/GoHiMCM

# 激活虚拟环境
source .venv/bin/activate

# 运行验证
python experiments/verify_parameters.py
```

### 方法 3: 单元测试

```python
import unittest
from src.environment.env import EvacuationEnv
from src.traditional_planner.scoring import PlannerConfig
from src.solutions.trad_v0.runner import run_planner

class TestParameterDeployment(unittest.TestCase):
    
    def test_office_params(self):
        env = EvacuationEnv(layout_name='office')
        config = PlannerConfig(alpha=0.15, beta=4.0, gamma=0.1)
        result = run_planner(env, config)
        
        self.assertGreater(result.efficiency, 0.2)
        self.assertEqual(result.rescued, 4)
    
    def test_universal_params(self):
        for layout in ['office', 'babycare', 'warehouse']:
            env = EvacuationEnv(layout_name=layout)
            config = PlannerConfig(alpha=0.15, beta=6.0, gamma=0.1)
            result = run_planner(env, config)
            
            self.assertGreater(result.efficiency, 0)
            self.assertGreater(result.rescued, 0)

if __name__ == '__main__':
    unittest.main()
```

---

## 🎯 分步骤部署示例

### 完整的现实例子

```python
"""
evacuation_runner.py - 完整的部署示例
"""

import sys
sys.path.append('/Users/hengshao/Desktop/HIMCM/GoHiMCM')

from src.environment.env import EvacuationEnv
from src.traditional_planner.scoring import PlannerConfig
from src.solutions.trad_v0.runner import run_planner


def main():
    """主函数 - 演示如何部署最优参数"""
    
    # 步骤 1: 确定地图
    layout = 'office'  # 或 'babycare', 'warehouse'
    
    # 步骤 2: 创建环境
    print(f"创建环境: {layout}")
    env = EvacuationEnv(layout_name=layout)
    
    # 步骤 3: 选择参数
    print(f"选择参数...")
    
    # ✅ 方案 1: 推荐 - 通用参数
    config = PlannerConfig(alpha=0.15, beta=6.0, gamma=0.1)
    
    # ❓ 方案 2: 特定地图 (取消注释使用)
    # if layout == 'office':
    #     config = PlannerConfig(alpha=0.15, beta=4.0, gamma=0.1)
    # elif layout == 'babycare':
    #     config = PlannerConfig(alpha=0.15, beta=2.0, gamma=0.2)
    # else:  # warehouse
    #     config = PlannerConfig(alpha=0.2, beta=2.0, gamma=0.1)
    
    print(f"参数: α={config.alpha}, β={config.beta}, γ={config.gamma}")
    
    # 步骤 4: 运行优化
    print(f"运行优化...")
    result = run_planner(env, config)
    
    # 步骤 5: 查看结果
    print(f"\n✅ 运行完成!")
    print(f"效率: {result.efficiency:.4f}")
    print(f"救援人数: {result.rescued}")
    print(f"完成时间: {result.time}")
    
    # 步骤 6: 验证效果
    if result.efficiency > 0.2:
        print("✅ 效率良好!")
    else:
        print("⚠️ 效率可能偏低,考虑调整参数")


if __name__ == '__main__':
    main()
```

运行:
```bash
python evacuation_runner.py
```

---

## 🔄 参数调优循环

如果部署后性能不理想，可以按以下流程调优：

### 流程图

```
开始
  ↓
选择参数
  ↓
运行测试
  ↓
检查效率
  ├→ 效率 < 目标? → 调整参数 ↺
  │              (见下表)
  └→ 效率 ≥ 目标? → 部署上线 ✅
```

### 参数调整指南

| 问题 | 原因 | 调整方案 |
|------|------|---------|
| 效率太低 | Planner 太保守 | 降低 β (6.0 → 2.0) |
| 效率太低 | 地图复杂 | 提高 α (0.15 → 0.20) |
| Agent 碰撞 | 拥堵权重不够 | 提高 γ (0.1 → 0.2 或 0.3) |
| 路径太长 | 距离权重过低 | 提高 α (0.15 → 0.20) |
| 救援困难 | 风险权重过高 | 降低 β (4.0 → 2.0) |

### 调参示例

```python
# 初始: 通用参数
config = PlannerConfig(alpha=0.15, beta=6.0, gamma=0.1)
result1 = run_planner(env, config)

if result1.efficiency < 0.2:
    # 效率太低，降低风险权重
    config = PlannerConfig(alpha=0.15, beta=2.0, gamma=0.1)
    result2 = run_planner(env, config)
    
    if result2.efficiency > result1.efficiency:
        print("✅ 调参成功!")
        config = result2的参数
```

---

## 📊 部署清单

部署前检查:

- [ ] 已导入 `PlannerConfig`
- [ ] 已导入 `run_planner`
- [ ] 已选择合适的参数
- [ ] 已创建 `EvacuationEnv`
- [ ] 已能运行基础测试

部署后检查:

- [ ] 代码能正常执行
- [ ] 结果符合预期
- [ ] 没有 error 或 warning
- [ ] 性能指标达到目标
- [ ] 文档已更新

---

## 💾 保存配置

### 方案 1: 保存到 Python 文件

```python
# saved_config.py
from src.traditional_planner.scoring import PlannerConfig

# Office 最优配置
OFFICE_CONFIG = PlannerConfig(
    alpha=0.15,
    beta=4.0,
    gamma=0.1
)

# Babycare 最优配置
BABYCARE_CONFIG = PlannerConfig(
    alpha=0.15,
    beta=2.0,
    gamma=0.2
)

# Warehouse 最优配置
WAREHOUSE_CONFIG = PlannerConfig(
    alpha=0.2,
    beta=2.0,
    gamma=0.1
)

# 通用配置 (推荐)
DEFAULT_CONFIG = PlannerConfig(
    alpha=0.15,
    beta=6.0,
    gamma=0.1
)
```

使用:
```python
from saved_config import OFFICE_CONFIG, DEFAULT_CONFIG

config = OFFICE_CONFIG  # 或 DEFAULT_CONFIG
```

### 方案 2: 保存到 JSON

```json
{
  "office": {
    "alpha": 0.15,
    "beta": 4.0,
    "gamma": 0.1
  },
  "babycare": {
    "alpha": 0.15,
    "beta": 2.0,
    "gamma": 0.2
  },
  "warehouse": {
    "alpha": 0.2,
    "beta": 2.0,
    "gamma": 0.1
  },
  "default": {
    "alpha": 0.15,
    "beta": 6.0,
    "gamma": 0.1
  }
}
```

使用:
```python
import json
from src.traditional_planner.scoring import PlannerConfig

with open('params.json', 'r') as f:
    params_dict = json.load(f)

config = PlannerConfig(**params_dict['office'])
```

---

## 🆘 常见部署问题

### Q1: ImportError: cannot import name 'PlannerConfig'

**解决**:
```python
# 确保导入路径正确
from src.traditional_planner.scoring import PlannerConfig

# 或检查 sys.path
import sys
sys.path.append('/Users/hengshao/Desktop/HIMCM/GoHiMCM')
```

### Q2: 结果和预期不符

**解决**:
```python
# 检查参数是否正确设置
print(f"α={config.alpha}, β={config.beta}, γ={config.gamma}")

# 对比优化结果中的参数值
# 查看 experiments/multi_layout_results/summary.json
```

### Q3: 效率远低于预期

**解决**:
```python
# 确认使用了正确的参数
# Office 最优: (0.15, 4.0, 0.1)
# Babycare 最优: (0.15, 2.0, 0.2)
# Warehouse 最优: (0.2, 2.0, 0.1)

# 如果仍然低，可能是:
# 1. 环境配置不同
# 2. 随机种子不同
# 3. 需要进一步微调
```

---

## ✨ 部署完成检查表

```bash
# 1. 环境检查
python -c "from src.traditional_planner.scoring import PlannerConfig; print('✅ Import OK')"

# 2. 基础测试
python -c "
from src.traditional_planner.scoring import PlannerConfig
config = PlannerConfig(0.15, 6.0, 0.1)
print(f'✅ Config created: α={config.alpha}, β={config.beta}, γ={config.gamma}')
"

# 3. 完整测试
python experiments/verify_parameters.py
```

---

## 📞 获取帮助

- 🔍 查看文档: `PARAMETER_OPTIMIZATION_RESULTS.md`
- ⚡ 快速参考: `PARAMETER_QUICK_REFERENCE.md`
- 📊 查看数据: `experiments/multi_layout_results/summary.json`
- 🎨 查看图表: `experiments/multi_layout_results/parameter_comparison.png`

---

**状态**: ✅ 准备就绪  
**版本**: 1.0  
**更新**: 2025-11-18
