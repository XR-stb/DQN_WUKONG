## 演示视频
https://www.bilibili.com/video/BV1DrpheREXh




## 使用 Conda

#### 1. 创建 Conda 环境
```shell
conda create --name wukong python=3.10
conda activate wukong
```

#### 2. 先通过 Conda 安装大包，再用 `pip` 安装其他库
为了避免兼容性和网络问题，你可以先使用 Conda 安装一些比较常见的库，再通过 `pip` 安装其余库：

```shell
# 方式一：使用 Conda 安装主要库
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 方式二：使用 pip 安装其余库
pip install --upgrade pip

pip install -r requirements.txt

# 方式三：或者使用 uv 安装： 
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# 然后配置环境变量： uv.exe
uv venv # 创建虚拟环境
uv pip install -r requirements.txt
uv run main.py
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

## 训练前准备工作
- 游戏设置：显示模式修改为窗口模式，分辨率调整为1280*720，你也可以在window.py中调整自己喜欢的窗口分辨率
- 设置游戏视角自动锁定boss
- 游戏窗口需要以**左上角为起始坐标**，不过我已经在你启动display_game_info.py脚本的时候做了这个事情, 你启动主流程时也会自动调整，可以按需调整
```
# 实际游戏窗口大小
game_width = 1280  # NOTE: 替换成你游戏的宽度和分辨率
game_height = 720
```
修改好分辨率后，运行下面的脚本，查看血量识别这些是否吻合
```
python -m utils.display_game_info
```

## 主要文件介绍
- window.py：画面中各血条和游戏窗口的矩形坐标定义
- judge.py: 奖励函数
- main.py: 程序入口
- train_data/*.py：训练数据可视化展示，结果生成在train_data/image目录下
- config/*.yaml：模型参数配置、动作空间配置

## 开始训练
启动脚本训练，根据提示按g可暂停和开始，启动脚本后，需要进入boss对战看到boss血条后，等待几秒即可启动模型。
```
python main.py 
```


## 部分代码和思路来自以下仓库，感谢他们
- https://github.com/analoganddigital/DQN_play_sekiro
- https://github.com/Sentdex/pygta5

## 更通用的框架
- https://github.com/Tencent/GameAISDK