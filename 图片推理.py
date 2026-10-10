import numpy as np
from PIL import Image
from vegetation_predictor import VegetationSegmenter

# 1. 启动推理引擎 (加载你之前训练好的 pkl)
segmenter = VegetationSegmenter("xgboost_stage2_model.pkl")

# 2. 读取一张全新的植物照片
print("正在读取图像...")
img = Image.open("test_leaf.jpg").convert("RGB")
img_array = np.array(img)  # 形状比如是 (1080, 1920, 3)

# 3. 展平图像：把 3D 的图像矩阵变成一长串 2D 的像素列表 (N, 3)
h, w, c = img_array.shape
flat_pixels = img_array.reshape(-1, 3)

# 4. 🚀 引擎轰鸣，开始推理！
print("正在进行像素级分割...")
preds, probs = segmenter.predict_pixels(flat_pixels, threshold=0.5)

# 5. 把一长串预测结果重新折叠回原图的形状 (1080, 1920)
mask_array = preds.reshape(h, w)

# 6. 生成黑白掩码图：绿色部分(1)变成白色(255)，枯黄部分(0)变成黑色(0)
mask_image = Image.fromarray((mask_array * 255).astype(np.uint8))
mask_image.save("segmentation_result.png")
print("搞定！掩码图已保存为 segmentation_result.png")