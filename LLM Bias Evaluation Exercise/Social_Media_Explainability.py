import pandas as pd
import numpy as np
import torch
import shap
import lime
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import re
import os
from tqdm import tqdm

# --- 配置 ---
# 使用您训练好的 SBIC 模型路径
MODEL_PATH = "../Model Training and Evaluation/model_output_sbic_albertv2"
# 您的社交媒体数据
DATA_PATH = "SBIC_full_results.csv"
# 输出文件
OUTPUT_FILE = "SBIC_full_results_Social_Media_Explainability_Results.csv"


# --- 1. 辅助函数 (修复了之前的 list/numpy 错误) ---
def compute_cosine_similarity(v1, v2):
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    if v1.size == 0 or v2.size == 0: return 0.0
    # Cosine distance to similarity
    return 1.0 - cosine(v1, v2) if np.any(v1) and np.any(v2) else 0.0


def compute_js_divergence(v1, v2):
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    # Convert to probability distributions
    v1 = np.abs(v1) / (np.sum(np.abs(v1)) + 1e-9)
    v2 = np.abs(v2) / (np.sum(np.abs(v2)) + 1e-9)
    return jensenshannon(v1, v2)


def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+', '', text)  # 去除链接
    text = re.sub(r'^RT\s+', '', text)
    return text.strip()


# --- 2. 预测封装函数 (供 LIME 使用) ---
def get_predict_proba_fn(pipe):
    def predict_proba(texts):
        # 强制返回 (N, 2) 的 numpy 数组
        results = pipe(texts, top_k=None, truncation=True, max_length=512)
        probs = []
        for res in results:
            # 排序确保顺序是 [LABEL_0, LABEL_1] (即 [无偏见, 有偏见])
            sorted_res = sorted(res, key=lambda x: x['label'])
            probs.append([x['score'] for x in sorted_res])
        return np.array(probs)

    return predict_proba


# --- 3. 主逻辑 ---
def main():
    # 检测 GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"Running on device: {'GPU' if device == 0 else 'CPU'}")

    # 加载模型
    print("Loading SBIC Model...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # 建立 Pipeline
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, top_k=None)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 加载数据
    print(f"Loading Data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("Data file not found. Please create Social_Media_Trends_2025.csv first.")
        return

    df = pd.read_csv(DATA_PATH)
    df['clean_text'] = df['clean_text'].apply(clean_text)

    # 为了演示，如果数据太多，只取前 20 条进行解释性分析（因为 SHAP 很慢）
    # 如果您有 GPU 且想跑完，可以注释掉下面这行
    sample_df = df.head(20).copy()
    print(f"Analyzing {len(sample_df)} samples...")

    # --- SHAP 分析 ---
    print("\nRunning SHAP Analysis (this may take time)...")
    # SHAP Explainer
    explainer = shap.Explainer(pipe)
    shap_values = explainer(sample_df['clean_text'].tolist())

    # --- LIME 分析 ---
    print("Running LIME Analysis...")
    lime_explainer = LimeTextExplainer(class_names=["Non-Biased", "Biased"])
    predict_fn = get_predict_proba_fn(pipe)

    results_data = []

    for i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        text = row['clean_text']
        platform = row.get('platform', 'Unknown')

        # 1. 获取 SHAP 值 (针对 'Biased' 类别，通常是 index 1)
        # shap_values[i] 是一个对象，.values 是 (token_len, num_classes)
        # 我们取所有 tokens 针对 Class 1 (Biased) 的贡献值
        current_shap_values = shap_values[i].values[:, 1]
        tokens = shap_values[i].data

        # 2. 获取 LIME 值
        exp = lime_explainer.explain_instance(text, predict_fn, num_features=len(tokens), labels=[1])
        # LIME 返回的是 (word, score) 列表，我们需要将其映射回原始 token 顺序
        lime_map = dict(exp.as_list(label=1))

        # 对齐 SHAP 和 LIME 的 Token
        # 注意：LIME 是基于词的，SHAP 是基于 Token 的，这里做简化对齐
        current_lime_values = []
        for token in tokens:
            # 简单去除空格来匹配 LIME 的词
            clean_token = token.strip().lower()
            # 尝试在 LIME 结果中找到该词，找不到则设为 0
            # 这是一个近似匹配，用于计算相似度
            val = next((v for k, v in lime_map.items() if k in clean_token or clean_token in k), 0.0)
            current_lime_values.append(val)

        current_lime_values = np.array(current_lime_values)

        # 3. 计算一致性指标 (Confidence Score)
        # 使用之前修复的函数
        cos_sim = compute_cosine_similarity(current_shap_values, current_lime_values)
        js_div = compute_js_divergence(current_shap_values, current_lime_values)

        # 4. 找出最重要的关键词 (Top Feature)
        # SHAP 中绝对值最大的 token
        top_token_idx = np.argmax(np.abs(current_shap_values))
        top_token = tokens[top_token_idx]

        # 获取模型预测结果
        pred_result = pipe(text)[0]
        # 排序找到得分最高的
        pred_label = sorted(pred_result, key=lambda x: x['score'], reverse=True)[0]
        is_biased = 1 if pred_label['label'] == 'LABEL_1' else 0
        confidence = pred_label['score']

        results_data.append({
            'platform': platform,
            'text': text,
            'prediction': 'Biased' if is_biased else 'Non-Biased',
            'model_confidence': confidence,
            'top_influential_token': top_token,  # 最能决定偏见的词
            'explanation_confidence_score': cos_sim,  # SHAP/LIME 一致性
            'js_divergence': js_div
        })

    # 保存结果
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Explainability Analysis Complete! Results saved to {OUTPUT_FILE}")
    print(results_df[['text', 'prediction', 'top_influential_token', 'explanation_confidence_score']].head())


if __name__ == "__main__":
    main()