## 演示视频
https://www.bilibili.com/video/BV1DrpheREXh




## 使用 Conda

#### 1. 创建 Conda 环境
```shell
conda create --name wukong python=3.10
conda activate wukong
```

#### 2. 先通过 Conda 安装大包，再用 `pip` 安装其他库
为了避免兼容性问题，你可以先使用 Conda 安装一些比较常见的库，再通过 `pip` 安装其余库：

```shell
# 使用 Conda 安装主要库
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 使用 pip 安装其余库
pip install --upgrade pip

pip install -r requirements.txt

```

#### 3. 检查安装结果
安装完成后，确认所有库是否正确安装：

```shell
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('GPU is', 'available' if torch.cuda.is_available() else 'not available')"

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

## 使用 BossRush V3 mod重复训练已打死的妖王
作者演示视频： https://www.bilibili.com/video/BV1QDxHeBETk
N站下载链接：https://www.nexusmods.com/blackmythwukong/mods/668

## 部分代码来自以下仓库，感谢开源
- https://github.com/analoganddigital/DQN_play_sekiro
- https://github.com/Sentdex/pygta5

## 更通用的框架
- https://github.com/Tencent/GameAISDK