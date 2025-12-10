import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report
import os
from tqdm import tqdm

# --- 配置 ---
# 模型路径 (您刚刚训练好的模型保存目录)
MODEL_PATH = "./model_output_sbic_albertv2"
# 测试数据路径 (SBIC 的测试集文件)
TEST_DATA_RAW = "SBIC.v2.agg.tst.csv"
# 输出目录
OUTPUT_DIR = "./result_output_sbic"


def clean_text(text):
    import re
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^RT\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def prepare_test_data():
    print(f"Loading test data from {TEST_DATA_RAW}...")
    if not os.path.exists(TEST_DATA_RAW):
        print(f"❌ Error: Test file not found {TEST_DATA_RAW}")
        return None

    df = pd.read_csv(TEST_DATA_RAW)

    # 预处理逻辑同训练时一致
    if 'hasBiasedImplication' in df.columns:
        df['label'] = (df['hasBiasedImplication'] == 0).astype(int)
    else:
        print("Warning: 'hasBiasedImplication' column not found. Labels might be incorrect.")
        df['label'] = 0  # Default if missing

    df['clean_text'] = df['post'].apply(clean_text)

    # 筛选有效数据
    df = df[['clean_text', 'label']].dropna()
    print(f"Loaded {len(df)} test samples.")
    return df


def evaluate():
    # 1. 准备数据
    test_df = prepare_test_data()
    if test_df is None:
        return

    # 2. 加载训练好的模型
    print(f"Loading fine-tuned model from {MODEL_PATH}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Please ensure that you have successfully run Train_SBIC_Model.py and that the model is saved in the correct location.")
        return

    # 3. 设置 GPU 推理
    device = 0 if torch.cuda.is_available() else -1
    print(f"Running inference on device: {'GPU' if device == 0 else 'CPU'}")

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=128
    )

    # 4. 批量预测
    print("Predicting...")
    predictions = []
    # 使用 batch_size=64 加速
    for out in tqdm(classifier(test_df['clean_text'].tolist(), batch_size=64), total=len(test_df)):
        # label 通常是 'LABEL_0' 或 'LABEL_1'
        label_id = int(out['label'].split('_')[-1])
        predictions.append(label_id)

    # 5. 生成报告
    print("\n" + "=" * 30)
    print("   SBIC MODEL EVALUATION REPORT")
    print("=" * 30)

    # 打印 Classification Report 到控制台
    report_dict = classification_report(test_df['label'], predictions, target_names=['Non-Biased', 'Biased'],
                                        output_dict=True)
    report_text = classification_report(test_df['label'], predictions, target_names=['Non-Biased', 'Biased'])
    print(report_text)

    # 6. 保存结果文件
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 保存 classification_report.csv
    report_df = pd.DataFrame(report_dict).transpose()
    report_save_path = os.path.join(OUTPUT_DIR, "classification_report.csv")
    report_df.to_csv(report_save_path)
    print(f"\n✅ The report has been saved. {report_save_path}")

    # 保存 full_results.csv (包含原文、真实标签、预测标签)
    test_df['predicted_label'] = predictions
    # 映射回可读标签
    label_map = {0: 'Non-Biased', 1: 'Biased'}
    test_df['actual_label_str'] = test_df['label'].map(label_map)
    test_df['predicted_label_str'] = test_df['predicted_label'].map(label_map)

    full_results_path = os.path.join(OUTPUT_DIR, "full_results.csv")
    test_df.to_csv(full_results_path, index=False)
    print(f"✅ Detailed results have been saved.: {full_results_path}")


if __name__ == "__main__":
    evaluate()