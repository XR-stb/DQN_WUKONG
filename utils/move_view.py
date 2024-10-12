import cv2
import pytesseract
import pyautogui
import numpy as np
import time

# 配置 tesseract 的路径（如果需要）
pytesseract.pytesseract.tesseract_cmd = r'D:\apps\Tesseract-OCR\tesseract.exe'
# testdata_dir_config = '--tessdata-dir "D:\apps\Tesseract-OCR\tessdata1"'

def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    processed_image = preprocess_image(screenshot)
    return processed_image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 反转颜色
    # inverted = cv2.bitwise_not(gray)
    
    # # 调整对比度和亮度
    # alpha = 1.5  # 对比度控制
    # beta = 0    # 亮度控制
    # adjusted = cv2.convertScaleAbs(inverted, alpha=alpha, beta=beta)
    
    # # 应用自适应阈值
    # binary = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_text_position(image, text, lang='chi_sim'):
    # data = pytesseract.image_to_data(image, lang=lang, config=testdata_dir_config, output_type=pytesseract.Output.DICT)
    data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT, config='--psm 6')
    log.debug(data)  # 打印识别结果
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        if text in data['text'][i]:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            return (x + w // 2, y + h // 2)  # 返回文字区域的中心点
    return None

def move_mouse_to_position(position):
    if position:
        pyautogui.moveTo(position)
        log.debug(f"Moved mouse to position: {position}")
    else:
        log.debug("Text not found.")

def main():
    # 定义截图区域 (left, top, width, height)
    region = (800, 450, 1300, 960)  # 你可以根据需要调整这些值

    # 截取屏幕
    image = capture_screen(region=region)

    # 显示预处理后的图像
    # cv2.imshow("Processed Screenshot", image)
    # cv2.waitKey(0)
    # cv2.destroyAllwindows()

    # 查找文字 "上香"
    position = find_text_position(image, "上香")

    # 移动鼠标到文字位置
    move_mouse_to_position(position)

if __name__ == "__main__":
    wait = 5
    for i in range(wait, 0, -1):
        log.debug('%d秒后复活' % i)
        time.sleep(1)
    main()