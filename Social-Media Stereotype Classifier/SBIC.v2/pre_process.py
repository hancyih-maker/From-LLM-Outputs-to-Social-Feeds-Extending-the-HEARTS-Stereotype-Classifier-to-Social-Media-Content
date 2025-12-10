import pandas as pd
import re
import os


# 1. 文本清洗函数
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 去除 "RT " 前缀 (Retweet标记)
    text = re.sub(r'^RT\s+', '', text)
    # 将多个空白字符（换行、制表符等）替换为单个空格
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# 2. SBIC 数据预处理函数
def preprocess_sbic_file(file_path):
    print(f"Processing: {file_path} ...")
    df = pd.read_csv(file_path)

    # --- 构建标签 (Label) ---
    # SBIC 聚合文件中: hasBiasedImplication=1 表示没有刻板印象(No Bias), 0 表示有(Has Bias)
    # 我们将其反转: 1 = 有偏见 (Biased), 0 = 无偏见 (Unbiased/Safe)
    if 'hasBiasedImplication' in df.columns:
        df['label'] = (df['hasBiasedImplication'] == 0).astype(int)
    else:
        # 如果读取的是非聚合文件，逻辑可能不同，这里默认处理聚合文件
        print("Warning: The 'hasBiasedImplication' column was not found. Please ensure you are using the 'agg' version file.")
        return None

    # --- 清洗文本 ---
    df['text'] = df['post'].apply(clean_text)

    # --- 辅助特征 ---
    # 保留 offensiveYN 数值作为辅助标签（可选，用于回归任务或加权）
    # offensiveYN > 0.5 通常也意味着有毒内容

    # 筛选需要的列
    processed_df = df[['text', 'label', 'offensiveYN', 'intentYN']]
    print(f"Processing complete, number of samples: {len(processed_df)}")
    print("Number of positive samples (biased):", processed_df['label'].sum())
    return processed_df


# 3. 处理您自己的真实社交媒体数据
def preprocess_real_data(file_path, text_column_name='text'):
    """
    用于处理您从社交媒体下载的真实数据
    """
    df = pd.read_csv(file_path)
    df['clean_text'] = df[text_column_name].apply(clean_text)
    # 真实数据没有标签，只返回清洗后的文本
    return df[['clean_text']]


# --- 执行预处理 ---

# 定义文件路径 (根据您上传的文件名)
files = {
    'train': 'SBIC.v2.agg.trn.csv',
    'dev': 'SBIC.v2.agg.dev.csv',
    'test': 'SBIC.v2.agg.tst.csv'
}

# 批量处理并保存
for split_name, file_name in files.items():
    if os.path.exists(file_name):
        df_processed = preprocess_sbic_file(file_name)
        if df_processed is not None:
            # 保存为新的 CSV，方便后续加载
            save_name = f"processed_sbic_{split_name}.csv"
            df_processed.to_csv(save_name, index=False)
            print(f"Saved to: {save_name}\n")
    else:
        print(f"The file does not exist.: {file_name}")

# 查看训练集前几行样例
if os.path.exists('processed_sbic_train.csv'):
    print("=== Training set sample data ===")
    print(pd.read_csv('processed_sbic_train.csv').head())
