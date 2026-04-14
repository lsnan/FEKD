import os
import cv2
import numpy as np

# -----------------------------
# 路径（按你的实际情况改）
# -----------------------------
pred_dir = "/media/user/b7c077e3-1e3a-4d1b-b7cc-5a47704b176e/lsn/code/BiRefNet-main_1/e_preds/BSL--student_fre_l34_70-epoch_70/COD10K-v3"
gt_dir   = "/home/user/lc/datasets/COD/COD10K-v3/Train/GT_Instance"
save_txt = "most_similar_result.txt"

# ------------------------------------
# IoU 计算函数 (二值化后计算)
# ------------------------------------
def compute_iou(pred, gt):
    pred_bin = (pred > 128).astype(np.uint8)
    gt_bin   = (gt > 128).astype(np.uint8)

    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    if union == 0:
        return 0.0
    return intersection / union


# ------------------------------------
# 搜索所有图像并记录 IoU
# ------------------------------------
records = []

for fname in os.listdir(pred_dir):

    pred_path = os.path.join(pred_dir, fname)
    gt_path   = os.path.join(gt_dir, fname)

    if not os.path.exists(gt_path):
        print(f"[WARN] GT 不存在: {gt_path}")
        continue

    pred = cv2.imread(pred_path, 0)
    gt   = cv2.imread(gt_path, 0)

    pred = cv2.resize(pred, gt.shape[::-1])  # 对齐大小

    iou = compute_iou(pred, gt)
    records.append((fname, iou))

    print(f"{fname}: IoU={iou:.4f}")

# ------------------------------------
# 找出最高 IoU 的图像
# ------------------------------------
records = sorted(records, key=lambda x: x[1], reverse=True)
best_fname, best_iou = records[0]

print("\n=============================")
print(" 最接近 GT 形状的预测结果：")
print("=============================")
print(f"文件名：{best_fname}")
print(f"IoU：{best_iou:.4f}")

with open(save_txt, "w") as f:
    f.write(f"most similar: {best_fname}, IoU={best_iou:.4f}\n")

print(f"\n结果已保存到 {save_txt}")
