import cv2
import numpy as np
import BloodCommon
import grabscreen

# 读取图像
image = grabscreen.grab_screen(BloodCommon.get_skill_2_window())

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化图像
_, binary = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 假设血条是最大的白色区域
max_contour = max(contours, key=cv2.contourArea)

# 获取血条的边界框
x, y, w, h = cv2.boundingRect(max_contour)

# 计算血量百分比
total_length = image.shape[1]  # 假设血条的总长度是图像的宽度
current_health_length = w
health_percentage = (current_health_length / total_length) * 100

print(f"Current Health: {health_percentage:.2f}%")

# 可视化结果
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Health Bar', image)
cv2.waitKey(0)
cv2.destroyAllWindows()