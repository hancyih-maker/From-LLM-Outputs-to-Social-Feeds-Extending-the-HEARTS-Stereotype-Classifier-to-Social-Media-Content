from transformers import pipeline
import numpy as np
import pandas as pd
import torch
import shap
from lime.lime_text import LimeTextExplainer
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

file_path = 'full_results_albertv2.csv'
model_path = "holistic-ai/bias_classifier_albertv2"

# Select the random sample of observations to use methods on
def sample_observations(file_path, k, seed):
    data = pd.read_csv(file_path)
    
    combinations = data.groupby(['dataset_name', 'categorisation'])
    
    sampled_data = pd.DataFrame(columns=data.columns)
    
    for name, group in combinations:
        same_label = group[group['predicted_label'] == group['actual_label']]
        diff_label = group[group['predicted_label'] != group['actual_label']]
        
        if len(same_label) >= k:
            same_sample = same_label.sample(n=k, random_state=seed)
        else:
            same_sample = same_label
        
        if len(diff_label) >= k:
            diff_sample = diff_label.sample(n=k, random_state=seed)
        else:
            diff_sample = diff_label
        
        sampled_data = pd.concat([sampled_data, same_sample, diff_sample], axis=0)
    
    sampled_data.reset_index(drop=True, inplace=True)
    
    print(sampled_data)
    
    return sampled_data

sampled_data = sample_observations(file_path, k=37, seed=42)
sampled_data.to_csv('sampled_data.csv')

# Define function to compute SHAP values
def shap_analysis(sampled_data, model_path):
    pipe = pipeline("text-classification", model=model_path, return_all_scores=True)
    masker = shap.maskers.Text(tokenizer=r'\b\w+\b')  
    explainer = shap.Explainer(pipe, masker)

    results = []
    class_names = ['LABEL_0', 'LABEL_1']
    
    for index, row in sampled_data.iterrows():
        text_input = [row['text']]
        shap_values = explainer(text_input)
        
        print(f"Dataset: {row['dataset_name']} - Categorisation: {row['categorisation']} - Predicted Label: {row['predicted_label']} - Actual Label: {row['actual_label']}")
        label_index = class_names.index("LABEL_1")  
        
        specific_shap_values = shap_values[:, :, label_index].values
        
        tokens = re.findall(r'\w+', row['text'])
        for token, value in zip(tokens, specific_shap_values[0]):
            results.append({
                'sentence_id': index, 
                'token': token, 
                'value_shap': value,
                'sentence': row['text'],
                'dataset': row['dataset_name'],
                'categorisation': row['categorisation'],
                'predicted_label': row['predicted_label'],
                'actual_label': row['actual_label']
            })
                
    return pd.DataFrame(results)


shap_results = shap_analysis(sampled_data, model_path)
print(shap_results)

# Define function to compute LIME values 
def custom_tokenizer(text):
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens

def lime_analysis(sampled_data, model_path):
    pipe = pipeline("text-classification", model=model_path, return_all_scores=True)
    
    def predict_proba(texts):
        preds = pipe(texts, return_all_scores=True)
        probabilities = np.array([[pred['score'] for pred in preds_single] for preds_single in preds])
        print("Probabilities shape:", probabilities.shape)
        return probabilities    
    
    explainer = LimeTextExplainer(class_names=['LABEL_0', 'LABEL_1'], split_expression=lambda x: custom_tokenizer(x))  
    
    results = []
    
    for index, row in sampled_data.iterrows():
        text_input = row['text']
        tokens = custom_tokenizer(text_input)
        exp = explainer.explain_instance(text_input, predict_proba, num_features=len(tokens), num_samples=100)
        
        print(f"Dataset: {row['dataset_name']} - Categorisation: {row['categorisation']} - Predicted Label: {row['predicted_label']} - Actual Label: {row['actual_label']}")

        explanation_list = exp.as_list(label=1)
        
        token_value_dict = {token: value for token, value in explanation_list}

        for token in tokens:
            value = token_value_dict.get(token, 0)  
            results.append({
                'sentence_id': index, 
                'token': token, 
                'value_lime': value,
                'sentence': text_input,
                'dataset': row['dataset_name'],
                'categorisation': row['categorisation'],
                'predicted_label': row['predicted_label'],
                'actual_label': row['actual_label']
            })

    return pd.DataFrame(results)

lime_results = lime_analysis(sampled_data, model_path)
print(lime_results)

lime_results.to_csv('lime_results.csv')
shap_results.to_csv('shap_results.csv')

# Define helper functions
# Define helper functions fixed
def compute_cosine_similarity(vector1, vector2):
    # === 修复：先转换为 numpy array ===
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    # 处理可能的空向量或维度不足情况（可选，增强鲁棒性）
    if v1.size == 0 or v2.size == 0:
        return 0.0

    return cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0][0]


def compute_pearson_correlation(vector1, vector2):
    # === 修复：先转换为 numpy array ===
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    # Pearson 需要至少两个数据点
    if len(v1) < 2 or len(v2) < 2:
        return 0.0

    correlation, _ = pearsonr(v1, v2)
    # 处理可能的 NaN 结果
    if np.isnan(correlation):
        return 0.0
    return correlation


def to_probability_distribution(values):
    # === 修复：先转换为 numpy array，并指定类型为 float ===
    values = np.array(values, dtype=float)

    min_val = np.min(values)
    if min_val < 0:
        values += abs(min_val)
    total = np.sum(values)
    if total > 0:
        values /= total
    else:
        # 如果总和为0，返回均匀分布以避免除以零错误
        values = np.ones_like(values) / len(values)
    return values


def compute_js_divergence(vector1, vector2):
    # 这里不需要显式转换，因为 to_probability_distribution 会处理
    # 但为了保险，可以加上 copy() 确保不修改原数据
    prob1 = to_probability_distribution(vector1)
    prob2 = to_probability_distribution(vector2)
    return jensenshannon(prob1, prob2)

shap_df = pd.read_csv('shap_results.csv')
lime_df = pd.read_csv('lime_results.csv')

# Compute similarity scores by token
token_shap = shap_df.groupby('token')['value_shap'].apply(list).reset_index()
token_lime = lime_df.groupby('token')['value_lime'].apply(list).reset_index()
token_merged = pd.merge(token_shap, token_lime, on='token', how='inner')
token_merged['cosine_similarity'] = token_merged.apply(lambda row: compute_cosine_similarity(row['value_shap'], row['value_lime']), axis=1)
token_merged['pearson_correlation'] = token_merged.apply(lambda row: compute_pearson_correlation(row['value_shap'], row['value_lime']), axis=1)
token_merged['js_divergence'] = token_merged.apply(lambda row: compute_js_divergence(row['value_shap'], row['value_lime']), axis=1)
token_merged.to_csv('token_level_similarity.csv')

# ==========================================
# Compute similarity scores by sentence
# ==========================================

# 1. 找出公共列用于合并
common_columns = [col for col in shap_df.columns if col != 'value_shap' and col != 'value_lime']

# 2. 合并 SHAP 和 LIME 结果，确保 Token 对齐
merged_df = pd.merge(shap_df, lime_df, on=common_columns, suffixes=('_shap', '_lime'))

# 3. 按句子 ID 分组，将每个句子的所有 Token 值聚合成列表
grouped = merged_df.groupby('sentence_id').agg({
    'value_shap': list,
    'value_lime': list
})

print("Computing sentence-level metrics...")

# 4. 【关键修复】使用 apply(axis=1) 逐行计算，而不是传入整列
# 注意：这里我们直接在 grouped（句子级数据）上计算指标
grouped['cosine_similarity'] = grouped.apply(lambda row: compute_cosine_similarity(row['value_shap'], row['value_lime']), axis=1)
grouped['pearson_correlation'] = grouped.apply(lambda row: compute_pearson_correlation(row['value_shap'], row['value_lime']), axis=1)
grouped['js_divergence'] = grouped.apply(lambda row: compute_js_divergence(row['value_shap'], row['value_lime']), axis=1)

# 5. 保存结果
output_file = 'sentence_level_similarity_results.csv'
grouped.to_csv(output_file)
print(f"Sentence-level results saved to {output_file}")
