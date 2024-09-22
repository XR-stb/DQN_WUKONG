import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image_path = "./images/screen.png"
image = cv2.imread(image_path)

# 定义全局变量
ref_point = []
cropping = False

# 鼠标回调函数
def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image

    # 如果左键按下，记录起始点并清除之前的矩形框
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # 如果左键释放，记录结束点并裁剪区域
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # 画出矩形（只允许存在一个矩形框）
        image_copy = cv2.imread(image_path)  # 重新加载图像，清除之前的矩形
        cv2.rectangle(image_copy, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image_copy)
        image = image_copy  # 更新图像

# 创建窗口并绑定鼠标事件
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# 显示图像，按 'q' 退出
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # 按下 'r' 重置图像
    if key == ord("r"):
        image = cv2.imread(image_path)

    # 按下 'c' 裁剪并计算灰度值
    elif key == ord("c"):
        if len(ref_point) == 2:
            roi = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]

            # 检查ROI是否为空或无效
            if roi.size == 0:
                print("Error: Selected region is empty. Please select a valid region.")
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 计算平均灰度值
            avg_gray_value = np.mean(gray_roi)

            # 显示裁剪的区域和灰度图
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 3, 1)
            plt.title("Cropped Image")
            plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            plt.subplot(1, 3, 2)
            plt.title("Grayscale")
            plt.imshow(gray_roi, cmap='gray')

            # 在第三个子图中显示平均灰度值
            plt.subplot(1, 3, 3)
            plt.title(f"Avg Gray Value: {avg_gray_value:.2f}")
            plt.text(0.5, 0.5, f"{avg_gray_value:.2f}", ha='center', va='center', fontsize=20)
            plt.axis('off')

            plt.show()

    # 按下 'q' 退出
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
