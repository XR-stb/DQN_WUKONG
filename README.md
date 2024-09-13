## 演示视频
https://www.bilibili.com/video/BV1DrpheREXh




## 使用 Conda

#### 1. 创建 Conda 环境
```shell
conda create --name game_ai python=3.10
conda activate game_ai
```

#### 2. 先通过 Conda 安装大包，再用 `pip` 安装其他库
为了避免兼容性问题，你可以先使用 Conda 安装一些比较常见的库，再通过 `pip` 安装其余库：

```shell
# 使用 Conda 安装主要库
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# 使用 pip 安装其余库
pip install --upgrade pip

# Anything above 2.10 is not supported on the GPU on Windows Native
pip install "numpy<2.0"
pip install "tensorflow<2.11" 

pip install -r requirements.txt
```

#### 3. 检查安装结果
安装完成后，确认所有库是否正确安装：

```shell
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print('GPU is', 'available' if tf.config.list_physical_devices('GPU') else 'not available')"

```



## 主要文件介绍
- window.py：画面各血条的矩形坐标定义
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