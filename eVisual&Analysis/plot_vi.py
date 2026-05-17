import json
import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_vi_distribution(json_path, output_image="vi_distribution.png"):
    """
    读取 VI_distribution.json，绘制 average_vi 的概率密度分布图。
    
    参数:
        json_path: JSON 文件路径
        output_image: 输出图片路径
    """
    # 读取数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取 average_vi 值
    vi_values = [entry["average_vi"] for entry in data]
    vi_values = np.array(vi_values)
    
    # 去除可能的异常值（比如负值，但这里都是正的）
    vi_values = vi_values[vi_values >= 0]
    
    if len(vi_values) == 0:
        print("没有有效的 VI 数据。")
        return
    
    # 创建图形
    plt.figure(figsize=(8, 5))
    
    # 绘制直方图（归一化，显示密度）
    counts, bins, patches = plt.hist(vi_values, bins=20, density=True, 
                                     alpha=0.6, color='steelblue', edgecolor='black', 
                                     label='Histogram')
    
    # 绘制核密度估计曲线
    try:
        kde = gaussian_kde(vi_values, bw_method='scott')
        x_range = np.linspace(min(vi_values), max(vi_values), 200)
        kde_vals = kde(x_range)
        plt.plot(x_range, kde_vals, 'r-', linewidth=2, label='KDE')
    except Exception as e:
        print(f"KDE 计算失败: {e}")
    
    # 设置轴标签和标题
    plt.xlabel('Average VI', fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title('Distribution of Variation of Information (VI)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 可选：添加均值/中位数垂直线
    mean_vi = np.mean(vi_values)
    median_vi = np.median(vi_values)
    plt.axvline(mean_vi, color='green', linestyle='--', linewidth=1.5, label=f'Mean: {mean_vi:.3f}')
    plt.axvline(median_vi, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {median_vi:.3f}')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"分布图已保存为 {output_image}")
    plt.show()

if __name__ == "__main__":
    # 请根据实际路径修改
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(base_dir, "VI_distribution.json")
    plot_vi_distribution(json_file, "vi_distribution_figure.png")