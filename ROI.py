import cv2, os
from natsort import natsorted

input_dir = r"C:/Users/user/AppData/Roaming/PotPlayerMini64/Capture/1012_1"
output_dir = r"C:/_Weifu/_master_ws/Tool_codes/ROI"
os.makedirs(output_dir, exist_ok=True)

img_files = natsorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))])
first_img = cv2.imread(os.path.join(input_dir, img_files[6]))

# 手動選一次 ROI
print("🟩 在第一張圖上畫 ROI，按 ENTER 確認。")
x, y, w, h = cv2.selectROI("Select ROI", first_img, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

# 批次裁切
for i, fname in enumerate(img_files):
    img = cv2.imread(os.path.join(input_dir, fname))
    roi = img[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(output_dir, f"{i:03d}_{fname}"), roi)

print(f"✅ ROI: x={x}, y={y}, w={w}, h={h}")
