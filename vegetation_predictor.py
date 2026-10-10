import pickle
import numpy as np
from PIL import Image
from skimage import color
import xgboost  # 必须导入，否则 pickle 加载时找不到引擎

def get_vegetation_features(pixels):
    """
    核心特征提取 (与训练时必须保持 100% 绝对一致)
    """
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8)

    pixels_float = pixels.astype(np.float32)
    r, g, b = pixels_float[:, 0], pixels_float[:, 1], pixels_float[:, 2]

    pixels_normalized = pixels_float / 255.0
    hsv = color.rgb2hsv(pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)
    lab = color.rgb2lab(pixels_normalized.reshape(-1, 1, 3)).reshape(-1, 3)

    h, s = hsv[:, 0] * 360.0, hsv[:, 1] * 100.0
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

class VegetationSegmenter:
    """
    植被分割推理引擎
    """
    def __init__(self, model_path):
        # 1. 初始化时加载一次模型，常驻内存
        print(f"正在加载模型: {model_path} ...")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        print("模型加载成功！")

    def predict_pixels(self, rgb_matrix, threshold=0.5):
        """
        对像素矩阵进行预测
        rgb_matrix: shape 为 (N, 3) 的 RGB 数组
        """
        # 2. 提取特征
        features = get_vegetation_features(rgb_matrix)
        
        # 3. 预测概率 (拿到它是绿色的概率)
        probs = self.model.predict_proba(features)[:, 1]
        
        # 4. 根据阈值生成 0/1 标签
        preds = (probs >= threshold).astype(np.uint8)
        
        return preds, probs