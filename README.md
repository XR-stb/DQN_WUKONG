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

## 使用venv
```
python -m venv .venv
# windows 运行activate 激活环境
\.venv\Scripts\activate.bat

# 安装依赖
pip install -r requirements.txt
```


## 主要文件介绍
- window.py：画面各血条的矩形坐标定义
- judge.py: 奖励分计算
- restart.py: 死亡后从土地庙自动走到boss的固定逻辑
- main.py: 程序入口

## 准备工作
游戏设置：显示模式修改为窗口模式，分辨率调整为1280*720，你也可以在window.py中调整自己喜欢的窗口分辨率
游戏窗口需要以**左上角为起始坐标**，我们已经在你启动display_game_info.py脚本的时候做了这个事情
```
# 实际游戏窗口大小
game_width = 1280  # NOTE: 替换成你游戏的宽度和分辨率
game_height = 720
```
修改好分辨率后，运行下面的脚本，查看血量识别这些是否吻合
```
python -m utils.display_game_info
```

修改合适的死亡自动寻路脚本
restart 死亡自动训练逻辑后面改成了由config.yaml配置控制，你可在actions_config.yaml中配置动作进行死亡寻路自动训练

或者改回原先的restart脚本方式

## 开始训练
启动脚本训练，根据提示按g可暂停和开始，启动脚本后，需要进入boss对战看到boss血条后，等待几秒即可启动模型。
```
python main.py 
```

## 使用 BossRush V3 mod重复训练已打死的妖王
作者演示视频： https://www.bilibili.com/video/BV1QDxHeBETk
N站下载链接：https://www.nexusmods.com/blackmythwukong/mods/668

## 部分代码来自以下仓库，感谢开源
- https://github.com/analoganddigital/DQN_play_sekiro
- https://github.com/Sentdex/pygta5

## 更通用的框架
- https://github.com/Tencent/GameAISDK