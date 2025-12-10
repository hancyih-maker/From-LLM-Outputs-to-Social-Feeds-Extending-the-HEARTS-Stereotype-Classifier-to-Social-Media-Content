import glob
import os
import numpy as np  # 导入 numpy 用于数值操作
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

def calculate_bias_and_aggregate():
    print("--- Summary of LLM Bias Assessment Results ---")

    results = []
    # 遍历当前目录下所有 *_predictions.csv 文件
    for filename in glob.glob("*_predictions.csv"):
        try:
            # 提取模型名称
            model_name = filename.replace(' Outputs_predictions.csv', '')

            # 读取数据
            df = pd.read_csv(filename)

            # <--- 修复 1：将字符串标签映射为数值 (1/0) --->
            # 假设标签是 'Stereotype' (刻板印象) 和 'Non-Stereotype' (非刻板印象)
            # 如果您的标签是 'LABEL_1'/'LABEL_0', 请修改映射字典
            label_mapping = {'Stereotype': 1, 'Non-Stereotype': 0}
            df['numerical_prediction'] = df['prediction'].map(label_mapping)

            # 检查是否有未映射的值
            if df['numerical_prediction'].isnull().any():
                # 如果有 NaN，可能是标签格式不匹配，强制将其视为 0
                df['numerical_prediction'] = df['numerical_prediction'].fillna(0)

            # 计算偏见得分 (求和 / 总数)
            total_samples = len(df)
            stereotype_count = df['numerical_prediction'].sum()
            bias_proportion = stereotype_count / total_samples

            results.append({
                'Model': model_name,
                'Total Samples': total_samples,
                'Stereotype Count': stereotype_count,
                'Bias Proportion': bias_proportion
            })

            print(f"✅ {model_name}: Bias Rate = {bias_proportion:.2%}")

            # 为下一步分组统计做准备，将 numerical_prediction 附着到总 DataFrame
            df['Model'] = model_name
            if 'df_all' not in locals():
                df_all = df
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

        except FileNotFoundError:
            print(f"❌ 处理文件 {filename} 时出错: 文件未找到.")
        except Exception as e:
            print(f"❌ 处理文件 {filename} 时出错: {type(e).__name__}: {e}")

    # 保存总汇总结果
    results_df = pd.DataFrame(results)
    output_summary_path = 'LLM_Bias_Summary.csv'
    results_df.to_csv(output_summary_path, index=False)
    print(f"\nThe summary results have been saved to {output_summary_path}")

    # --- 修复 2：按群体分组的平均偏见比例计算 (避免 TypeError) ---
    print("\n--- Average bias percentage by group ---")
    if 'df_all' in locals():
        # 对数值化后的列进行平均值计算 (mean of 1s and 0s is the proportion)
        group_summary = df_all.groupby('group')['numerical_prediction'].mean().sort_values(ascending=False)
        print(group_summary)

        # 额外保存分组统计结果
        group_summary.to_csv('LLM_Bias_Group_Summary.csv', header=['Bias Proportion'])
    else:
        print("没有找到可用于分组统计的数据。")


if __name__ == "__main__":
    calculate_bias_and_aggregate()



# 1. 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# 2. 定义模型发布日期 (硬编码数据，来源自论文及公开信息)
model_dates = {
    'Claude-2': '2023-07-11',
    'Claude-3-Sonnet': '2024-02-29', # 实际上是 Opus 发布日，Sonnet 同期
    'Claude-3.5-Sonnet': '2024-06-20',
    'Gemini-1.0-Pro': '2024-02-08', # 广泛可用日期
    'Gemini-1.5-Pro': '2024-04-09', # 公开预览
    'GPT-3.5-Turbo': '2022-11-30', # ChatGPT 发布日
    'GPT-4-Turbo': '2023-11-06',
    'GPT-4o': '2024-05-13',
    'Llama-3-70B-T': '2024-04-18',
    'Llama-3.1-405B-T': '2024-07-23',
    'Mistral Large 2': '2024-07-24',
    'Mistral Medium': '2023-12-11'
}

# 3. 读取你计算出的汇总数据
try:
    df = pd.read_csv('LLM_Bias_Summary.csv')
except FileNotFoundError:
    print("❌ 未找到 LLM_Bias_Summary.csv，请先运行上一步的统计脚本！")
    exit()

# 4. 数据处理：合并日期
df['Release Date'] = df['Model'].map(model_dates)
df['Release Date'] = pd.to_datetime(df['Release Date'])
df['Bias Percentage'] = df['Bias Proportion'] * 100  # 转换为百分比

# 移除没有日期数据的模型（防止报错）
df = df.dropna(subset=['Release Date'])

# 5. 开始绘图
plt.figure(figsize=(14, 8))

# 创建散点图
# 使用 hue='Model' 给每个点不同的颜色
scatter = sns.scatterplot(
    data=df,
    x='Release Date',
    y='Bias Percentage',
    hue='Model',
    s=200, # 点的大小
    alpha=0.9,
    palette='tab20' # 颜色盘
)

# 6. 添加文本标签 (防止重叠是个难点，这里做简单偏移)
for i, row in df.iterrows():
    plt.text(
        row['Release Date'],
        row['Bias Percentage'] + 0.8, # 稍微向上偏移
        row['Model'],
        fontsize=9,
        ha='center',
        fontweight='bold',
        color='black'
    )

# 7. 添加红色虚线 (基准线，例如论文中的平均水平)
# 这里我们画一条平均线作为参考
avg_bias = df['Bias Percentage'].mean()
plt.axhline(y=avg_bias, color='r', linestyle='--', label=f'Average Bias ({avg_bias:.1f}%)')

# 8. 格式化图表
plt.title('Stereotype Proportion by Model Release Date (Replication)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Stereotype Proportion (%)', fontsize=12)
plt.xlabel('Model Release Date', fontsize=12)
plt.ylim(30, 65) # 根据数据范围调整 Y 轴

# 格式化 X 轴日期
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=0)

# 移除图例 (因为标签已经在点旁边了)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

# 9. 保存图片
output_img = 'Figure_5_Replication.png'
plt.tight_layout()
plt.savefig(output_img, dpi=300)
print(f"✅ The chart has been generated and saved as: {output_img}")
plt.show()