import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def get_vegetation_features(pixels):
    """
    核心特征工程：从 RGB 像素中提取 14 维颜色空间特征，用于区分健康与枯黄植被。
    """
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8)

    pixels_float = pixels.astype(np.float32)
    r, g, b = pixels_float[:, 0], pixels_float[:, 1], pixels_float[:, 2]

    # 转换到 HSV 和 LAB 色彩空间
    pixels_normalized = pixels_float / 255.0
    hsv = color.rgb2hsv(pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)
    lab = color.rgb2lab(pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)

    h = hsv[:, 0] * 360.0
    s = hsv[:, 1] * 100.0
    a, bb = lab[:, 1], lab[:, 2]
    gray = np.mean(pixels_float, axis=1)

    # 提取类似 CMYK 和 YCbCr/YIQ 的色彩分量
    c, m_raw, y_raw = 1.0 - r / 255.0, 1.0 - g / 255.0, 1.0 - b / 255.0
    k = np.minimum.reduce([c, m_raw, y_raw])
    denom = 1.0 - k + 1e-8
    m = ((m_raw - k) / denom) * 100.0
    ye = ((y_raw - k) / denom) * 100.0

    pil_line = Image.fromarray(pixels.reshape(-1, 1, 3), mode="RGB")
    ycbcr = np.array(pil_line.convert("YCbCr"), dtype=np.float32).reshape(-1, 3)
    cb, cr = ycbcr[:, 1], ycbcr[:, 2]

    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b

    # 组合成 14 维特征矩阵
    features = np.column_stack([r, g, b, h, s, a, bb, gray, m, ye, cb, cr, i, q])
    return np.nan_to_num(features)

def load_and_preprocess_data(csv_path, green_label, senescent_label, ignore_label):
    """
    读取 CSV 数据，清洗无效标签，并提取特征。
    """
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    valid_rows = []
    labels = []
    
    # 过滤无效数据，统一下游的 0/1 标签
    for row in rows:
        label = int(float(row["Y"]))
        if label == ignore_label:
            continue
        if label in (green_label, senescent_label):
            valid_rows.append(row)
            labels.append(1 if label == green_label else 0)

    # 提取 RGB 矩阵并计算特征
    rgb_matrix = np.array([[int(r["R"]), int(r["G"]), int(r["B"])] for r in valid_rows], dtype=np.uint8)
    features = get_vegetation_features(rgb_matrix)
    labels = np.array(labels, dtype=np.uint8)

    return features, labels, valid_rows

def train_and_evaluate(args):
    print("🚀 开始加载和处理数据...")
    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据准备
    X, y, rows = load_and_preprocess_data(
        csv_path, args.green_label, args.senescent_label, args.ignore_label
    )
    print(f"✅ 数据加载完成，共有效样本: {len(y)} 个。特征维度: {X.shape}")

    # 2. 划分训练集与测试集 (支持区组划分或纯随机划分)
    if rows and "REP" in rows[0]:
        print(f"📌 检测到 REP 区组信息，按区组 {args.rep_train} (训练) / {args.rep_test} (测试) 划分。")
        train_idx = [i for i, r in enumerate(rows) if str(r["REP"]).upper() == args.rep_train.upper()]
        test_idx = [i for i, r in enumerate(rows) if str(r["REP"]).upper() == args.rep_test.upper()]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        print("🎲 未检测到 REP 区组，进行 75% / 25% 随机分层抽样划分。")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

    # 计算类别权重，防止由于绿叶或枯叶过多导致样本不平衡
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    sample_weights = np.array([total_samples / (2.0 * class_counts[label]) for label in y_train])

    # 3. 初始化并训练 XGBoost 引擎 (适配最新版 XGBoost)
    print("🌲 开始训练 XGBoost 模型 (这可能需要一些时间)...")
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",  # 使用直方图算法，在大数据集上极速训练
        random_state=42,
        n_jobs=-1,           # 火力全开，调用服务器所有 CPU 核心
    )
    
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # 4. 模型评估
    print("📊 正在评估模型表现...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy (准确率)": float(accuracy_score(y_test, preds)),
        "F1 Score - Green (健康组织 F1)": float(f1_score(y_test, preds, pos_label=1)),
        "F1 Score - Senescent (衰老组织 F1)": float(f1_score(y_test, preds, pos_label=0)),
        "ROC AUC (曲线下面积)": float(roc_auc_score(y_test, probs)),
        "Confusion Matrix (混淆矩阵)": confusion_matrix(y_test, preds).tolist(),
    }

    # 5. 保存结果
    model_path = output_dir / "xgboost_stage2_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    report_path = output_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"🎉 训练完成！模型已保存至: {model_path}")
    print("📝 模型测试集得分如下:")
    print(json.dumps(metrics, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="植被表型分割 - XGBoost 纯净训练脚本")
    parser.add_argument("--csv", required=True, help="输入的数据集 CSV 路径")
    parser.add_argument("--output-dir", default="artifacts", help="模型和报告保存的文件夹")
    
    # 标签定义
    parser.add_argument("--green-label", type=int, default=1, help="健康植被的标签值")
    parser.add_argument("--senescent-label", type=int, default=0, help="衰老/枯黄植被的标签值")
    parser.add_argument("--ignore-label", type=int, default=2, help="需要忽略的背景标签值")
    
    # 区组划分设置
    parser.add_argument("--rep-train", default="TRAIN", help="训练区组的名称")
    parser.add_argument("--rep-test", default="TEST", help="测试区组的名称")
    
    # XGBoost 核心超参数
    parser.add_argument("--n-estimators", type=int, default=300, help="决策树的数量")
    parser.add_argument("--max-depth", type=int, default=6, help="树的最大深度")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="学习率")

    args = parser.parse_args()
    train_and_evaluate(args)