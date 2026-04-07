import cv2
from pathlib import Path

src_dir = Path("/home/njust/Fire/Data/DataTest_02/MV-CS004-10UC(DA7064688)/no_smokefire")   # bmp 所在文件夹
dst_dir = Path("/home/njust/Fire/Data/DataTest_02/MV-CS004-10UC(DA7064688)/no_smokefire_01")   # 输出 jpg 文件夹
dst_dir.mkdir(exist_ok=True)

for bmp_path in src_dir.glob("*.bmp"):
    img = cv2.imread(str(bmp_path))
    if img is None:
        print("读取失败：", bmp_path)
        continue

    jpg_path = dst_dir / (bmp_path.stem + ".jpg")
    # 质量 95，基本看不出压缩损失
    cv2.imwrite(str(jpg_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
