## 演示视频
https://www.bilibili.com/video/BV1DrpheREXh

## 安装必要的库

```shell
pip install numpy
pip install pillow
pip install opencv-python
pip install pywin32
pip install pandas
pip install gym

pip install tensorflow
pip install tensorflow-gpu
 
pip install numpy opencv-python pillow pywin32 gym pandas
pip install tensorflow

pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple # tensorflow装不了的话用清华源会快一点

# 如果需要ocr来定位视角就安装
pip install pyautogui pytesseract
https://github.com/UB-Mannheim/tesseract/wiki
```

## cuda 安装
选择系统适合的cuda版本: https://developer.nvidia.com/cuda-toolkit-archive
```
C:\Users\Administrator>nvidia-smi
Sat Aug 24 22:21:03 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
```
安装cudann 加速训练过程：
https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.4.0.58_cuda12-archive.zip

## 主要文件介绍
- BloodCommon.py：画面各血条的矩形坐标定义
- judge.py: 奖励分计算
- restart.py: 死亡后从土地庙自动走到boss的固定逻辑
- training.py: 训练脚本，按 T 暂停或恢复训练

## 开始训练
1. 确认血条坐标是否和我一致，不一致则自己截图替换一下坐标即可
2. 启动脚本
```
python training.py
```

## 部分代码来自以下仓库，感谢开源
- https://github.com/analoganddigital/DQN_play_sekiro
- https://github.com/Sentdex/pygta5

## 更通用的框架
- https://github.com/Tencent/GameAISDK