import argparse
import csv
import json
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from skimage import color
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, XGBClassifier

# ==========================================
# 1. 常量与特征定义
# ==========================================
FEATURE_NAMES = [
    "R", "G", "B", "H", "S", "Lab_a", "Lab_b", 
    "Gray", "M", "Y", "Cb", "Cr", "I", "Q"
]

# ==========================================
# 2. 核心特征工程
# ==========================================
def get_vegetation_features(pixels):
    """从 RGB 像素中提取 14 维颜色空间特征"""
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8)

    pixels_float = pixels.astype(np.float32)
    r, g, b = pixels_float[:, 0], pixels_float[:, 1], pixels_float[:, 2]

    pixels_normalized = pixels_float / 255.0
    hsv = color.rgb2hsv(pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)
    lab = color.rgb2lab(pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)

    h = hsv[:, 0] * 360.0
    s = hsv[:, 1] * 100.0
    a, bb = lab[:, 1], lab[:, 2]
    gray = np.mean(pixels_float, axis=1)

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

    features = np.column_stack([r, g, b, h, s, a, bb, gray, m, ye, cb, cr, i, q])
    return np.nan_to_num(features)

# ==========================================
# 3. 数据加载与预处理
# ==========================================
def load_and_preprocess_data(csv_path, green_label, senescent_label, ignore_label):
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    valid_rows = []
    labels = []
    
    for row in rows:
        label = int(float(row["Y"]))
        if label == ignore_label:
            continue
        if label in (green_label, senescent_label):
            valid_rows.append(row)
            labels.append(1 if label == green_label else 0)

    rgb_matrix = np.array([[int(r["R"]), int(r["G"]), int(r["B"])] for r in valid_rows], dtype=np.uint8)
    features = get_vegetation_features(rgb_matrix)
    labels = np.array(labels, dtype=np.uint8)

    return features, labels, valid_rows

# ==========================================
# 4. SHAP 计算与绘图辅助函数
# ==========================================
def compute_xgboost_shap_values(model, features):
    """利用 XGBoost 原生接口快速计算 SHAP 值"""
    dmatrix = DMatrix(features, feature_names=FEATURE_NAMES)
    contributions = model.get_booster().predict(dmatrix, pred_contribs=True)
    return contributions[:, :-1]

def style_closed_box(ax, linewidth=0.8, color="#666666"):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_color(color)
    ax.tick_params(width=linewidth, color=color)

def build_density_jitter(values, rng, max_jitter=0.28):
    """为 SHAP 蜂群图生成抖动偏移量"""
    values = np.asarray(values, dtype=np.float32)
    count = len(values)
    if count <= 1:
        return np.zeros(count, dtype=np.float32)
    span = float(np.max(values) - np.min(values))
    if span < 1e-9:
        return rng.uniform(-max_jitter, max_jitter, size=count).astype(np.float32)
    bin_count = min(70, max(12, int(np.sqrt(count))))
    bins = np.linspace(np.min(values), np.max(values), bin_count + 1)
    bin_ids = np.clip(np.digitize(values, bins) - 1, 0, bin_count - 1)
    counts = np.bincount(bin_ids, minlength=bin_count).astype(np.float32)
    local_density = counts[bin_ids]
    local_density /= max(float(np.max(local_density)), 1.0)
    jitter_scale = 0.08 + local_density * (max_jitter - 0.08)
    offsets = rng.normal(loc=0.0, scale=jitter_scale / 2.6, size=count)
    return np.clip(offsets, -jitter_scale, jitter_scale).astype(np.float32)

# ==========================================
# 5. 可视化绘图函数
# ==========================================
# ==========================================
# 5. 可视化绘图函数 (已修改为生成单独的图片)
# ==========================================
def plot_results(output_dir, y_test, preds, probs, shap_values, X_test):
    """生成并分别保存各项可视化图表"""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------- 图 1: 混淆矩阵 (Confusion Matrix) ----------------
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_ratio = cm / np.where(row_sums == 0, 1, row_sums)
    ax.imshow(cm_ratio, cmap="YlGn", vmin=0.0, vmax=1.0)
    ax.set_title(f"Confusion Matrix (Acc: {accuracy_score(y_test, preds):.4f})", pad=12, fontweight='bold')
    ax.set_xticks([0, 1], labels=["Pred: Senescent", "Pred: Green"])
    ax.set_yticks([0, 1], labels=["True: Senescent", "True: Green"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{cm_ratio[i, j]*100:.1f}%", ha="center", va="center",
                    color="black" if cm_ratio[i, j] < 0.55 else "white", fontweight='bold')
    style_closed_box(ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "xgboost_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图 2: ROC 曲线 (ROC Curve) ----------------
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_value = roc_auc_score(y_test, probs)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax.plot(fpr, tpr, label=f"XGBoost (AUC={auc_value:.3f})", color="#2166ac", linewidth=2)
    ax.set_title("ROC Curve", pad=12, fontweight='bold')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(color="#d9d9d9", linewidth=0.6, alpha=0.45)
    style_closed_box(ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "xgboost_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图 3: 概率分布直方图 (Probability Distribution) ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(probs[y_test == 0], bins=bins, alpha=0.78, color="#df9a2b", label="True Senescent")
    ax.hist(probs[y_test == 1], bins=bins, alpha=0.60, color="#56ad72", label="True Green")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Probability Distribution", pad=12, fontweight='bold')
    ax.set_xlabel("Predicted Probability of Green")
    ax.set_ylabel("Pixel Count")
    ax.legend()
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.5)
    style_closed_box(ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "xgboost_probability_dist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图 4: SHAP 平均绝对值条形图 (Top 10) ----------------
    fig, ax = plt.subplots(figsize=(8, 6))
    shap_scores = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(shap_scores)[-10:]
    ax.barh([FEATURE_NAMES[i] for i in order], shap_scores[order], color="#347eb8", edgecolor="none")
    ax.set_title("Top 10 Features (Mean |SHAP|)", pad=12, fontweight='bold')
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    style_closed_box(ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "xgboost_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 图 5: SHAP 蜂群图 (Beeswarm) ----------------
    fig, ax = plt.subplots(figsize=(10, 8))
    rng = np.random.default_rng(42)
    max_points = 2400
    if shap_values.shape[0] > max_points:
        selected = np.sort(rng.choice(shap_values.shape[0], size=max_points, replace=False))
        s_vals, f_vals = shap_values[selected], X_test[selected]
    else:
        s_vals, f_vals = shap_values, X_test

    mean_abs = np.mean(np.abs(s_vals), axis=0)
    order = np.argsort(mean_abs)[::-1][:12] # 最多显示 12 个特征
    
    y_positions = np.arange(len(order))[::-1].astype(np.float32)
    cmap = mpl.colormaps["coolwarm"]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    for row_idx, feature_idx in enumerate(order):
        y0 = y_positions[row_idx]
        row_shap = s_vals[:, feature_idx]
        row_values = f_vals[:, feature_idx]
        
        low, high = np.percentile(row_values, 5), np.percentile(row_values, 95)
        scaled_vals = np.full_like(row_values, 0.5) if high - low < 1e-8 else np.clip((row_values - low) / (high - low), 0.0, 1.0)
        
        offsets = build_density_jitter(row_shap, rng)
        order_within = np.argsort(np.abs(row_shap))
        
        ax.scatter(row_shap[order_within], y0 + offsets[order_within], c=scaled_vals[order_within], 
                   cmap=cmap, norm=norm, s=15, alpha=0.6, linewidths=0)

    ax.axvline(0.0, color="#9a9a9a", linewidth=1.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=11)
    ax.set_xlabel("SHAP value (Impact on Model Output)", fontsize=12)
    ax.set_title("SHAP Beeswarm Plot", pad=12, fontweight='bold', fontsize=14)
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.6, alpha=0.75)
    style_closed_box(ax)
    
    fig.tight_layout()
    fig.savefig(plot_dir / "xgboost_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"🖼️ 可视化图表已分别独立保存至: {plot_dir}")


# ==========================================
# 6. 主训练逻辑
# ==========================================
def train_and_evaluate(args):
    print("🚀 开始加载和处理数据...")
    csv_path = Path(args.csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, rows = load_and_preprocess_data(csv_path, args.green_label, args.senescent_label, args.ignore_label)
    print(f"✅ 数据加载完成，有效样本: {len(y)} 个。特征维度: {X.shape}")

    if rows and "REP" in rows[0]:
        print(f"📌 检测到 REP 区组，按区组 {args.rep_train} / {args.rep_test} 划分。")
        train_idx = [i for i, r in enumerate(rows) if str(r["REP"]).upper() == args.rep_train.upper()]
        test_idx = [i for i, r in enumerate(rows) if str(r["REP"]).upper() == args.rep_test.upper()]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        print("🎲 未检测到 REP 区组，进行 75% / 25% 随机分层抽样划分。")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    class_counts = Counter(y_train)
    sample_weights = np.array([len(y_train) / (2.0 * class_counts[label]) for label in y_train])

    print("🌲 开始训练 XGBoost 模型...")
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist", 
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    print("📊 正在评估并生成可视化与 SHAP 解释...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # 计算 SHAP
    shap_values = compute_xgboost_shap_values(model, X_test)
    
    # 绘制图表
    plot_results(output_dir, y_test, preds, probs, shap_values, X_test)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "F1 Score (Green)": float(f1_score(y_test, preds, pos_label=1)),
        "F1 Score (Senescent)": float(f1_score(y_test, preds, pos_label=0)),
        "ROC AUC": float(roc_auc_score(y_test, probs)),
    }

    model_path = output_dir / "xgboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    with open(output_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print(f"🎉 训练与可视化全部完成！模型保存在: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost 表型分类与 SHAP 可视化")
    
    # 重点修改了这里的 default 值：
    parser.add_argument("--csv", default="data.csv", help="数据集 CSV 路径")
    parser.add_argument("--output-dir", default="result2", help="输出文件夹")
    
    parser.add_argument("--green-label", type=int, default=1)
    parser.add_argument("--senescent-label", type=int, default=0)
    parser.add_argument("--ignore-label", type=int, default=2)
    parser.add_argument("--rep-train", default="TRAIN")
    parser.add_argument("--rep-test", default="TEST")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)

    args = parser.parse_args()
    train_and_evaluate(args)