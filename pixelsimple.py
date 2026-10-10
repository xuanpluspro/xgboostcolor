import pickle
import numpy as np
from PIL import Image
from skimage import color

def get_vegetation_features(pixels):
    """
    【特征提取保持不变】
    从 RGB 像素中提取 14 维颜色空间特征。
    """
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


def direct_pixel_inference(model_path):
    # 1. 加载模型
    print("📦 正在加载 XGBoost 模型...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ==========================================
    # 2. 构造输入内容 (模拟 5 个代表性的像素点)
    # ==========================================
    # 格式要求: 形状为 (N, 3) 的二维 NumPy 数组，数据类型为 uint8
    # N 代表像素点的个数，3 代表 R, G, B 三个通道
    input_pixels = np.array([
        [34, 139, 34],   # 像素 0: 典型的健康深绿色
        [173, 255, 47],  # 像素 1: 嫩绿色 (黄绿色)
        [218, 165, 32],  # 像素 2: 典型的枯黄色 (衰老组织)
        [139, 69, 19],   # 像素 3: 泥土的深褐色 (模拟干扰项)
        [200, 190, 140]  # 像素 4: 极度干枯发白的叶片颜色
    ], dtype=np.uint8)
    
    print(f"📥 【输入格式】: {type(input_pixels)}, 形状: {input_pixels.shape}")
    print(f"📥 【输入内容】 (前三个):\n{input_pixels[:3]}\n")

    # 3. 提取特征
    features = get_vegetation_features(input_pixels)
    print(f"⚙️ 【特征格式】: 提取出形状为 {features.shape} 的 14 维特征矩阵\n")

    # ==========================================
    # 4. 模型预测 (输出内容)
    # ==========================================
    # predict() 输出类别 (0 或 1)
    # predict_proba() 输出属于各个类别的概率，[:, 1] 取出属于 1 (健康) 的概率
    preds = model.predict(features)
    probs = model.predict_proba(features)[:, 1]

    print(f"📤 【输出格式 (预测类别)】: {type(preds)}, 形状: {preds.shape}")
    print(f"📤 【输出内容 (预测类别)】: {preds}")
    print(f"📤 【输出内容 (健康概率)】: {np.round(probs, 4)}\n")

    # 5. 打印易读的结果报告
    print("📊 ============ 推理结果汇总 ============")
    class_map = {1: "健康植被 (Green)", 0: "衰老/枯黄 (Senescent)"}
    
    for idx, (pixel, pred, prob) in enumerate(zip(input_pixels, preds, probs)):
        r, g, b = pixel
        pred_label = class_map[pred]
        print(f"像素 {idx} [R:{r:<3} G:{g:<3} B:{b:<3}] -> 预测: {pred_label:<15} | 健康概率: {prob*100:>5.2f}%")

if __name__ == "__main__":
    # 假设你的模型在 artifacts 文件夹下
    model_path = "artifacts/xgboost_stage2_model.pkl"
    # 直接运行即可 (如果在真实环境下，确保路径正确)
    direct_pixel_inference(model_path)