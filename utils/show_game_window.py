import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import time
import sys
import window


image_scale = 1.3

# 读取图像
image_path = "./images/screen.png"
image = cv2.imread(image_path)


# 查找log位置
if not window.set_windows_offset(image):
    print("Failed to find the game logo.")               
    sys.exit()

image = window.game_window.color.copy()

origin_image = image

# 定义全局变量
ref_point = []
cropping = False
cropped_image = None  # 用于存储裁剪图像

# 鼠标回调函数
def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image, cropped_image

    # 如果左键按下，记录起始点并清除之前的矩形框
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # 如果左键释放，记录结束点并裁剪区域
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # 画出矩形（只允许存在一个矩形框）
        image_copy = origin_image.copy()  # 重新加载图像，清除之前的矩形
        cv2.rectangle(image_copy, ref_point[0], ref_point[1], (0, 255, 0), 1)
        cv2.imshow("image", image_copy)

        roi = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        cropped_image = roi 

        image = image_copy  # 更新图像


# 获取图像的高度和宽度
height, width = image.shape[:2]

# 创建可调整大小的窗口
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# 等比例放大窗口
cv2.resizeWindow("image", int(width * image_scale), int(height * image_scale))

# 绑定鼠标事件
cv2.setMouseCallback("image", click_and_crop)


# 显示图像，按 'q' 退出
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # 按下 'r' 重置图像
    if key == ord("r"):
        image = origin_image.copy()

    # 按下 'c' 裁剪并计算灰度值和亮度值
    elif key == ord("c"):
        if len(ref_point) == 2:
            roi = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            
            # 检查ROI是否为空或无效
            if roi.size == 0:
                print("Error: Selected region is empty. Please select a valid region.")
                continue
            
            # 转换为灰度图
            gray_roi = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            # 转换为HSL模型
            hsl_roi = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HLS)
            l_channel = hsl_roi[:, :, 1]  # 提取亮度通道

            # 计算平均灰度值
            avg_gray_value = np.mean(gray_roi)

            # 计算平均亮度值
            avg_luminance_value = np.mean(l_channel)

            # 获取起始点和结束点坐标
            start_point = ref_point[0]
            end_point = ref_point[1]

            # 显示裁剪的区域、灰度图和亮度图
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 4, 1)
            plt.title("Cropped Image")
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

            plt.subplot(1, 4, 2)
            plt.title("Grayscale")
            plt.imshow(gray_roi, cmap='gray')

            plt.subplot(1, 4, 3)
            plt.title("Luminance (HSL)")
            plt.imshow(l_channel, cmap='gray')

            # 在第四个子图中显示平均灰度值、平均亮度值和矩形坐标
            plt.subplot(1, 4, 4)
            plt.title(f"Avg Gray & Luminance")
            plt.text(0.5, 0.6, f"Start: {start_point}", ha='center', va='center', fontsize=12)
            plt.text(0.5, 0.5, f"End: {end_point}", ha='center', va='center', fontsize=12)
            plt.text(0.5, 0.3, f"Avg Gray: {avg_gray_value:.2f}", ha='center', va='center', fontsize=12)
            plt.text(0.5, 0.2, f"Avg Luminance: {avg_luminance_value:.2f}", ha='center', va='center', fontsize=12)
            plt.axis('off')

            plt.show()

    # 按下 's' 保存裁剪后的原图
    elif key == ord("s"):
        if cropped_image is not None:
            # 生成带时间戳的文件名
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = f"./images/cropped_image_{timestamp}.png"

            # 保存图像
            cv2.imwrite(save_path, cropped_image)
            print(f"Cropped image saved at {save_path}")
        else:
            print("No cropped image to save. Please select and crop a region first.")

    # 按下 'q' 退出
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
