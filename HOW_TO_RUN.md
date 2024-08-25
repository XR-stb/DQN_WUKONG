## 安装必要的库

```shell
pip install numpy
pip install pillow
pip install opencv-python
pip install pywin32
pip install pandas
pip install tensorflow
pip install gym
pip install tensorflow-gpu

pip install numpy opencv-python pillow pywin32 gym tensorflow tensorflow-gpu
```

## 生成训练数据
```shell
python ./image_grab.py
```

## 数据标准化
```shell
python ./data_test.py
```

## 开始训练
```shell
python ./train_data.py
```

## 看一下实际效果
```shell
python ./test_train.py
```

## 用新版本的脚本
```
# 看一下系统适合的cuda版本: https://developer.nvidia.com/cuda-toolkit-archive
C:\Users\Administrator>nvidia-smi
Sat Aug 24 22:21:03 2024
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |


pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
python .\DQN_sekiro_training_gpu.py
```
