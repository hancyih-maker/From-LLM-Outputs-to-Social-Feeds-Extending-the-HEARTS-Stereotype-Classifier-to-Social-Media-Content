import os
import pandas as pd

# 你已经有的 BASE_DIRS，如果前面没有就加上
BASE_DIRS = [
    "result_output_LR_tfidf",
    "result_output_LR_embedding",
]

def find_all_reports():
    report_paths = []

    for base_dir in BASE_DIRS:
        if not os.path.exists(base_dir):
            continue

        for root, dirs, files in os.walk(base_dir):
            if "classification_report.csv" in files:
                full_path = os.path.join(root, "classification_report.csv")
                report_paths.append(full_path)

    return report_paths


def collect_metrics():
    report_paths = find_all_reports()
    rows = []

    for path in report_paths:
        # 把 Windows 的反斜杠统一成标准格式
        norm_path = os.path.normpath(path)
        parts = norm_path.split(os.sep)

        # 预期结构：
        # [0] result_output_LR_tfidf / result_output_LR_embedding
        # [1] <trained_name>  e.g. mgsd_trained
        # [2] <evaluated_name> e.g. mgsd
        # [3] classification_report.csv
        if len(parts) < 4:
            print(f"Unexpected path structure, skip: {path}")
            continue

        result_root = parts[0]         # e.g. result_output_LR_tfidf
        trained_name = parts[1]        # e.g. mgsd_trained
        eval_name = parts[2]           # e.g. mgsd

        # 1) feature_type
        if "tfidf" in result_root:
            feature_type = "tfidf"
        elif "embedding" in result_root:
            feature_type = "embedding"
        else:
            feature_type = "unknown"

        # 2) trained_on 及其简化版本
        trained_on = trained_name                      # 原始名字
        trained_dataset = trained_name.replace("_trained", "")  # 去掉 _trained 后缀

        # 3) evaluated_on
        evaluated_on = eval_name

        # 4) 读 classification_report.csv
        try:
            df_report = pd.read_csv(path, index_col=0)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        # 保护性检查：必须要有 'macro avg' 和 'accuracy' 行
        if "macro avg" not in df_report.index or "accuracy" not in df_report.index:
            print(f"Report format unexpected, skip: {path}")
            continue

        # 提取指标
        macro_precision = df_report.loc["macro avg", "precision"]
        macro_recall = df_report.loc["macro avg", "recall"]
        macro_f1 = df_report.loc["macro avg", "f1-score"]

        # sklearn 的 classification_report 中：
        # 'accuracy' 这一行的 'f1-score' 列其实就是 accuracy 数值
        accuracy = df_report.loc["accuracy", "f1-score"]

        row = {
            "feature_type": feature_type,
            "trained_on": trained_on,
            "trained_dataset": trained_dataset,
            "evaluated_on": evaluated_on,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "report_path": path,  # 方便以后回溯
        }

        rows.append(row)

    # 汇总成一个 DataFrame
    metrics_df = pd.DataFrame(rows)

    # 排个序：先按 feature_type，再按 trained_dataset/evaluated_on
    metrics_df = metrics_df.sort_values(
        by=["feature_type", "trained_dataset", "evaluated_on"]
    ).reset_index(drop=True)

    # 保存到一个总表里
    output_path = "all_classification_metrics.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"\nSaved metrics summary to: {output_path}")

    # 顺便打印前几行看一下
    print("\nPreview of collected metrics:")
    print(metrics_df.head())


if __name__ == "__main__":
    collect_metrics()
