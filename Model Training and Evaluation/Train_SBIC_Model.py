import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# --- 配置参数 (复用 HEARTS 论文的最佳实践 ) ---
MODEL_NAME = "albert-base-v2"  # 使用论文推荐的低碳模型
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
EPOCHS = 6
MAX_LENGTH = 128  # 社交媒体文本通常较短


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


class SBICDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def run_training():
    print(f"Loading data...")
    # 读取预处理后的数据 (确保路径正确)
    # 如果您在根目录运行，可能需要调整路径，例如 '../processed_sbic_train.csv'
    try:
        df_train = pd.read_csv("processed_sbic_train.csv")
        df_dev = pd.read_csv("processed_sbic_dev.csv") if os.path.exists("processed_sbic_dev.csv") else None
    except FileNotFoundError:
        print("错误：找不到 processed_sbic_train.csv，请先运行预处理代码。")
        return

    # 数据清洗检查
    df_train = df_train.dropna(subset=['text', 'label'])
    df_train['text'] = df_train['text'].astype(str)
    df_train['label'] = df_train['label'].astype(int)


    # Tokenization
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = SBICDataset(train_encodings, train_labels)
    val_dataset = SBICDataset(val_encodings, val_labels)

    # 加载模型
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 将模型移动到 GPU (如果可用)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./results_sbic_albert',  # 输出目录
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=LEARNING_RATE,
        fp16=torch.cuda.is_available(),  # 如果是 GPU，开启混合精度加速
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 保存最终模型
    save_path = "./model_output_sbic_albertv2"
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Done!")


if __name__ == "__main__":
    run_training()