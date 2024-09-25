import yaml
from enum import Enum, auto
import threading
import time
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import atexit

# 动作执行器类
class ActionExecutor:
    def __init__(self, config_file):
        """初始化动作执行器，并从配置文件加载动作"""
        self.action_queue = []  # 动作队列
        self.interrupt_event = threading.Event()  # 中断事件
        self.keyboard = KeyboardController()  # 键盘控制器
        self.mouse = MouseController()  # 鼠标控制器
        self.running = True  # 控制执行器的运行状态
        self.pressed_keys = set()  # 记录按下的按键
        self.pressed_buttons = set()  # 记录按下的鼠标按钮
        self.action_finished_callback = None  # 动作完成后的回调
        self.thread = threading.Thread(target=self._execute_actions)
        self.thread.start()
        self.currently_executing = False  # 是否有动作在执行
        atexit.register(self.stop)  # 程序退出时自动停止

        # 从 YAML 配置文件中加载
        self.config = self.load_config(config_file)
        self.Action = self.create_action_enum(self.config['actions'])  # 动态创建枚举
        self.action_configs = self.config['actions']
        self.hot_list = self.config['hot_list']

    @staticmethod
    def load_config(file_path):
        """从 YAML 文件中加载配置"""
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def create_action_enum(actions_dict):
        """动态生成动作的枚举类型"""
        return Enum('Action', {name: auto() for name in actions_dict})

    def add_action(self, action_sequence, action_finished_callback=None):
        """添加动作到队列"""
        self.action_queue.append(action_sequence)
        self.action_finished_callback = action_finished_callback
        self.currently_executing = True  # 动作开始执行

    def _execute_actions(self):
        """顺序执行动作队列中的动作"""
        while self.running:
            if self.action_queue:
                action_sequence = self.action_queue.pop(0)
                self._run_action_sequence(action_sequence)

            time.sleep(0.01)  # 防止占用过多CPU

    def _run_action_sequence(self, action_sequence):
        """执行一个完整的动作序列"""
        for action in action_sequence:
            if self.interrupt_event.is_set():
                break  # 中断当前动作序列

            self._handle_action(action)

        # 动作完成后调用回调通知外部
        if not self.interrupt_event.is_set() and self.action_finished_callback:
            self.action_finished_callback()

        self.currently_executing = False  # 动作完成

    def _handle_action(self, action):
        """处理单个动作"""
        action_type = action[0]

        if action_type == 'press':
            self._press_key(action[1])
        elif action_type == 'release':
            self._release_key(action[1])
        elif action_type == 'press_mouse':
            self._press_mouse(action[1])
        elif action_type == 'release_mouse':
            self._release_mouse(action[1])
        elif action_type == 'move_mouse':
            self._move_mouse(action[1], action[2], action[3])
        elif action_type == 'delay':
            self._delay(action[1])

    def _press_key(self, key):
        """按下键盘按键并记录"""
        self.keyboard.press(key)
        self.pressed_keys.add(key)  # 记录按下的键

    def _release_key(self, key):
        """释放键盘按键并从记录中移除"""
        if key in self.pressed_keys:
            self.keyboard.release(key)
            self.pressed_keys.remove(key)

    def _press_mouse(self, button):
        """按下鼠标按钮并记录"""
        self.mouse.press(Button.left if button == 'left' else Button.right)
        self.pressed_buttons.add(button)  # 记录按下的鼠标按钮

    def _release_mouse(self, button):
        """释放鼠标按钮并从记录中移除"""
        if button in self.pressed_buttons:
            self.mouse.release(Button.left if button == 'left' else Button.right)
            self.pressed_buttons.remove(button)

    def _move_mouse(self, x_offset, y_offset, duration):
        """移动鼠标，持续一定时间"""
        start_time = time.time()
        while time.time() - start_time < duration:
            if self.interrupt_event.is_set():
                break  # 如果打断信号被设置，停止移动
            self.mouse.move(x_offset, y_offset)
            time.sleep(0.01)

    def _delay(self, duration):
        """延迟一段时间"""
        start_time = time.time()
        while time.time() - start_time < duration:
            if self.interrupt_event.is_set():
                break  # 如果打断信号被设置，提前结束延迟
            time.sleep(0.01)

    def interrupt_action(self):
        """打断当前正在执行的动作，并释放所有已按下的按键和鼠标按钮"""
        self.interrupt_event.set()
        self._release_all_pressed()  # 释放所有已按下的键和鼠标按钮
        self.currently_executing = False  # 动作被打断，标记为不再执行

    def _release_all_pressed(self):
        """释放所有已按下的按键和鼠标按钮"""
        for key in list(self.pressed_keys):
            self.keyboard.release(key)
        self.pressed_keys.clear()

        for button in list(self.pressed_buttons):
            self.mouse.release(Button.left if button == 'left' else Button.right)
        self.pressed_buttons.clear()

    def stop(self):
        """停止动作执行器"""
        self.running = False
        self.interrupt_event.set()
        self._release_all_pressed()  # 停止时释放所有按下的键和按钮
        self.thread.join()

    def take_action(self, action, action_finished_callback=None):
        """通过动作名称或索引执行动作，并设置动作完成后的回调"""
        if isinstance(action, int):  # 如果传入的是索引，从热列表中取动作
            if 0 <= action < len(self.hot_list):
                action_name = self.hot_list[action]
                action_sequence = self.action_configs.get(action_name)
            else:
                print(f"Invalid action index: {action}")
                return
        elif isinstance(action, str):  # 如果传入的是动作名称，直接取
            action_sequence = self.action_configs.get(action)
        else:
            print(f"Invalid action type: {type(action)}")
            return

        if action_sequence:
            self.add_action(action_sequence, action_finished_callback=action_finished_callback)
        else:
            print(f"Action not found: {action}")

    def is_running(self):
        """检查当前是否有动作在执行"""
        return self.currently_executing


# 外部接口示例
if __name__ == "__main__":
    # 初始化动作执行器
    executor = ActionExecutor('./config/actions_config.yaml')

    # 动作结束时的回调函数
    def on_action_finished():
        print("动作执行完毕")

    # 示例：通过 DQN 网络获取 action_id 并执行对应动作，并注册动作完成回调
    action_id = 0  # 假设 DQN 网络输出的 action_id 是 0
    executor.take_action(action_id, action_finished_callback=on_action_finished)  # 通过索引执行动作

    # 轮询检查是否在执行中
    while executor.is_running():
        print("动作正在执行中...")
        time.sleep(0.5)

    # 中途打断动作
    time.sleep(1)  # 假设 1 秒后需要打断动作
    executor.interrupt_action()  # 调用打断函数
