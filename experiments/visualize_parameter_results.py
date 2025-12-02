"""
参数优化结果可视化对比
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_parameter_comparison_plots():
    """创建参数优化结果的多个可视化"""
    
    results = {
        "office": {"alpha": 0.15, "beta": 4.0, "gamma": 0.1, "efficiency": 0.2619, "rescued": 4.0},
        "babycare": {"alpha": 0.15, "beta": 2.0, "gamma": 0.2, "efficiency": 0.5779, "rescued": 37.0},
        "warehouse": {"alpha": 0.2, "beta": 2.0, "gamma": 0.1, "efficiency": 0.0931, "rescued": 4.67},
    }
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 参数对比雷达图
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    layouts = list(results.keys())
    angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]
    
    for layout in layouts:
        values = [
            results[layout]['alpha'],
            results[layout]['beta'] / 10,  # 标准化到 0-1
            results[layout]['gamma'],
        ]
        values += values[:1]
        ax1.plot(angles, values, 'o-', linewidth=2, label=layout.capitalize())
        ax1.fill(angles, values, alpha=0.15)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(['Alpha', 'Beta (÷10)', 'Gamma'])
    ax1.set_ylim(0, 1)
    ax1.set_title('参数配置对比 (雷达图)', fontsize=12, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax1.grid(True)
    
    # 2. 效率对比
    ax2 = plt.subplot(2, 3, 2)
    layouts_names = ['Office', 'Babycare', 'Warehouse']
    efficiencies = [results[l]['efficiency'] for l in layouts]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax2.bar(layouts_names, efficiencies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('效率 (Rescued/Time)', fontsize=11, fontweight='bold')
    ax2.set_title('救援效率对比', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(efficiencies) * 1.2)
    
    # 添加数值标签
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{eff:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. 救援人数对比
    ax3 = plt.subplot(2, 3, 3)
    rescued = [results[l]['rescued'] for l in layouts]
    population = [4, 39, 48]
    rescue_pct = [rescued[i]/population[i]*100 for i in range(3)]
    
    x = np.arange(len(layouts_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, rescued, width, label='已救援', color='#95E1D3', edgecolor='black', linewidth=1.5)
    bars2 = ax3.bar(x + width/2, population, width, label='总人口', color='#F38181', edgecolor='black', linewidth=1.5)
    
    ax3.set_ylabel('人数', fontsize=11, fontweight='bold')
    ax3.set_title('救援人数对比', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(layouts_names)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Alpha 参数对比
    ax4 = plt.subplot(2, 3, 4)
    alphas = [results[l]['alpha'] for l in layouts]
    bars = ax4.barh(layouts_names, alphas, color='#FFB6B9', edgecolor='black', linewidth=2)
    ax4.set_xlabel('Alpha 值 (距离权重)', fontsize=11, fontweight='bold')
    ax4.set_title('Alpha 参数对比', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 0.3)
    
    for bar, alpha in zip(bars, alphas):
        width = bar.get_width()
        ax4.text(width, bar.get_y() + bar.get_height()/2.,
                f'{alpha:.2f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Beta 参数对比
    ax5 = plt.subplot(2, 3, 5)
    betas = [results[l]['beta'] for l in layouts]
    bars = ax5.barh(layouts_names, betas, color='#FEC8D8', edgecolor='black', linewidth=2)
    ax5.set_xlabel('Beta 值 (风险权重)', fontsize=11, fontweight='bold')
    ax5.set_title('Beta 参数对比', fontsize=12, fontweight='bold')
    ax5.set_xlim(0, 8)
    
    for bar, beta in zip(bars, betas):
        width = bar.get_width()
        ax5.text(width, bar.get_y() + bar.get_height()/2.,
                f'{beta:.1f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax5.grid(axis='x', alpha=0.3)
    
    # 6. Gamma 参数对比
    ax6 = plt.subplot(2, 3, 6)
    gammas = [results[l]['gamma'] for l in layouts]
    bars = ax6.barh(layouts_names, gammas, color='#FFDDC1', edgecolor='black', linewidth=2)
    ax6.set_xlabel('Gamma 值 (拥堵权重)', fontsize=11, fontweight='bold')
    ax6.set_title('Gamma 参数对比', fontsize=12, fontweight='bold')
    ax6.set_xlim(0, 0.4)
    
    for bar, gamma in zip(bars, gammas):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f'{gamma:.1f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax6.grid(axis='x', alpha=0.3)
    
    plt.suptitle('🎯 Greedy Sweep Planner - 最优参数优化结果对比', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path('/Users/hengshao/Desktop/HIMCM/GoHiMCM/experiments/multi_layout_results')
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(str(output_dir / 'parameter_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 参数对比图已保存: {output_dir / 'parameter_comparison.png'}")
    
    # 创建参数热力图
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    param_names = ['Alpha', 'Beta', 'Gamma']
    layout_names_short = ['Office', 'Babycare', 'Warehouse']
    
    for idx, param_name in enumerate(param_names):
        param_values = []
        for layout in layouts:
            if param_name == 'Alpha':
                param_values.append(results[layout]['alpha'])
            elif param_name == 'Beta':
                param_values.append(results[layout]['beta'])
            else:  # Gamma
                param_values.append(results[layout]['gamma'])
        
        ax = axes[idx]
        bars = ax.bar(layout_names_short, param_values, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                     edgecolor='black', linewidth=2, alpha=0.7)
        
        ax.set_ylabel('参数值', fontsize=11, fontweight='bold')
        ax.set_title(f'{param_name} 参数', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值
        for bar, val in zip(bars, param_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.suptitle('📊 参数值详细对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'parameter_detailed.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 详细参数图已保存: {output_dir / 'parameter_detailed.png'}")
    
    plt.close('all')
    print("\n✨ 所有可视化完成!")


if __name__ == "__main__":
    create_parameter_comparison_plots()
