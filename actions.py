# actions.py
import yaml
from enum import Enum, auto
import threading
import traceback
import time
import keys
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import atexit
from log import log

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
        self.action_executed_event = threading.Event()
        self.action_executed_event.set()  # 初始状态为已设置，表示执行器空闲
        self.thread = threading.Thread(target=self._execute_actions, daemon=True)
        self.thread.start()

        atexit.register(self.stop)  # 程序退出时自动停止

        #pynput的 鼠标移动 在游戏里不起作用 还是只能用老办法
        self.keys = keys.Keys()

        # 从 YAML 配置文件中加载
        self.config = self.load_config(config_file)
        self.Action = self.create_action_enum(self.config['actions'])  # 动态创建枚举
        self.action_configs = self.config['actions']
        self.hot_list = self.config['hot_list']
        self.no_interrupts_set = set(self.config['no_interrupts'])

    @staticmethod
    def load_config(file_path):
        """从 YAML 文件中加载配置"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    @staticmethod
    def create_action_enum(actions_dict):
        """动态生成动作的枚举类型"""
        return Enum('Action', {name: auto() for name in actions_dict})

    def add_action(self, action_sequence, action_finished_callback=None):
        # 等待执行器空闲
        if not self.action_executed_event.wait(timeout=5.0):
            log.debug("add_action: 等待执行器空闲超时")
            return  
        """添加动作到队列"""
        self.action_queue.append(action_sequence)
        self.action_finished_callback = action_finished_callback
        self.action_executed_event.clear()  # 清除事件，表示执行器正忙

    def _execute_actions(self):
        """从队列中执行动作。"""
        while self.running:
            if self.action_queue:
                action_sequence = self.action_queue.pop(0)
                self.interrupt_event.clear()
                try:
                    self._run_action_sequence(action_sequence)
                except Exception as e:
                    log.debug(f"_execute_actions 中的异常: {e}")
                    traceback.print_exc()
                finally:
                    self.action_executed_event.set()
            else:
                self.action_executed_event.set()
            time.sleep(0.001)

    def _flatten_action_sequence(self, action_sequence):
        """将嵌套的动作序列展平成单一列表"""
        flattened = []
        for action in action_sequence:
            if isinstance(action, list) and all(isinstance(item, list) for item in action):
                # 如果这个动作是一个完整的嵌套动作列表，递归展开它
                flattened.extend(self._flatten_action_sequence(action))
            else:
                # 否则这是一个正常的动作
                flattened.append(action)
        return flattened


    def _run_action_sequence(self, action_sequence):
        """执行一个完整的动作序列"""
        for action in self._flatten_action_sequence(action_sequence):
            if self.interrupt_event.is_set():
                break  # 中断当前动作序列

            self._handle_action(action)


        # 动作完成后调用回调通知外部
        if not self.interrupt_event.is_set() and self.action_finished_callback:
            self.action_finished_callback()


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
        elif action_type == 'move_mouse_absolute':
            self._move_mouse_absolute(action[1], action[2], action[3])
        elif action_type == 'delay':
            self._delay(action[1])

    def _press_key(self, key):
        """按下键盘按键并记录"""
        if self.interrupt_event.is_set():
            return
        # 判断 key 是否为特殊键
        if key in Key.__members__:  # 如果 key 是 'shift', 'ctrl', 'alt' 等
            self.keyboard.press(Key[key])  # 使用 Key 枚举类来处理特殊键
        else:
            self.keyboard.press(key)  # 对于普通字符，直接按下
        self.pressed_keys.add(key)  # 记录按下的键

    def _release_key(self, key):
        """释放键盘按键并从记录中移除"""
        if self.interrupt_event.is_set():
            return        
        if key in Key.__members__:
            self.keyboard.release(Key[key])  # 释放特殊键
        else:
            self.keyboard.release(key)  # 释放普通字符键
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)

    def _press_mouse(self, button):
        """按下鼠标按钮并记录"""
        if self.interrupt_event.is_set():
            return
        if button == 'left':
            self.mouse.press(Button.left)
        elif button == 'right':
            self.mouse.press(Button.right)
        elif button == 'middle':
            self.mouse.press(Button.middle)  # 处理鼠标中键
        else:
            raise ValueError(f"Unknown mouse button: {button}")
        
        self.pressed_buttons.add(button)  # 记录按下的鼠标按钮

    def _release_mouse(self, button):
        """释放鼠标按钮并从记录中移除"""
        if self.interrupt_event.is_set():
            return
        if button == 'left':
            self.mouse.release(Button.left)
        elif button == 'right':
            self.mouse.release(Button.right)
        elif button == 'middle':
            self.mouse.release(Button.middle)  # 处理鼠标中键
        else:
            raise ValueError(f"Unknown mouse button: {button}")
        
        if button in self.pressed_buttons:
            self.pressed_buttons.remove(button)

    def _move_mouse(self, x_offset, y_offset, duration):
        """在 duration 时间内平滑地移动鼠标，总移动距离为 x_offset 和 y_offset"""
        steps = int(duration / 0.01)  # 将移动过程拆分成多个小步
        x_step = int(x_offset / steps)  # 将 x 的偏移量转换为整数
        y_step = int(y_offset / steps)  # 将 y 的偏移量转换为整数

        start_time = time.perf_counter()
        for step in range(steps):
            if self.interrupt_event.is_set():
                break  # 如果打断信号被设置，停止移动
            #self.mouse.move(x_step, y_step)  # 相对移动
            self.keys.directMouse(x_step, y_step)

            # 计算下一步应该执行的时间点
            next_time = start_time + (step + 1) * (duration / steps)
            current_time = time.perf_counter()
            sleep_time = next_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 如果时间已经超出，直接进入下一步
                continue

        # 确保在最后完成全部偏移
        remaining_x = x_offset - x_step * steps
        remaining_y = y_offset - y_step * steps
        #self.mouse.move(remaining_x, remaining_y)
        self.keys.directMouse(x_step, y_step)

    def _move_mouse_absolute(self, target_x, target_y, duration):
        """在 duration 时间内平滑地移动鼠标到绝对坐标 (target_x, target_y)"""
        start_x, start_y = self.mouse.position  # 获取鼠标的当前绝对坐标
        steps = int(duration / 0.01)  # 将移动过程拆分成多个小步
        x_step = (target_x - start_x) / steps  # 每步 x 的移动量
        y_step = (target_y - start_y) / steps  # 每步 y 的移动量

        start_time = time.perf_counter()
        for step in range(steps):
            if self.interrupt_event.is_set():
                break  # 如果打断信号被设置，停止移动
            # 计算每步的目标位置，并设置鼠标位置为绝对坐标
            new_x = start_x + x_step * (step + 1)
            new_y = start_y + y_step * (step + 1)
            self.mouse.position = (new_x, new_y)  # 设置绝对坐标

            # 计算下一步应该执行的时间点
            next_time = start_time + (step + 1) * (duration / steps)
            current_time = time.perf_counter()
            sleep_time = next_time - current_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                continue

        # 最后确保鼠标准确移动到目标绝对坐标
        self.mouse.position = (target_x, target_y)

    def _delay(self, duration):
        """延迟一段时间"""

        start_time = time.perf_counter()
        end_time = start_time + duration
        while True:
            if self.interrupt_event.is_set():
                break  # 如果打断信号被设置，提前结束延迟
            current_time = time.perf_counter()
            remaining_time = end_time - current_time
            if remaining_time <= 0:
                break  # 延迟时间结束
            elif remaining_time > 0.01:
                time.sleep(0.01)  # 等待较长的时间，减少CPU占用
            else:
                # 等待剩余的时间
                time.sleep(remaining_time)

    def interrupt_action(self, timeout=20.0):
        """打断当前正在执行的动作，并释放所有已按下的按键和鼠标按钮"""
        ret = True
        self.interrupt_event.set()
        self.action_queue.clear()

        # 使用标记判断是否还在执行

        if not self.action_executed_event.wait(timeout=timeout):
            log.debug(f"Interrupt timed out after {timeout} seconds.")
            ret = False

        self._release_all_pressed()  # 释放所有已按下的键和鼠标按钮
        time.sleep(0.01)

        return ret

    def _release_all_pressed(self):
        """释放所有已按下的按键和鼠标按钮"""
        for key in list(self.pressed_keys):
            if key in Key.__members__:
                self.keyboard.release(Key[key])
            else:
                self.keyboard.release(key)
        self.pressed_keys.clear()

        for button in list(self.pressed_buttons):
            if button == 'left':
                self.mouse.release(Button.left)
            elif button == 'right':
                self.mouse.release(Button.right)
            elif button == 'middle':
                self.mouse.release(Button.middle)
        self.pressed_buttons.clear()

    def stop(self):
        """停止动作执行器"""
        self.running = False
        self.interrupt_event.set()
        self._release_all_pressed()  # 停止时释放所有按下的键和按钮
        self.thread.join()

    def get_action_size(self):
        return len(self.hot_list)

    def get_action_name(self,index):
        return self.hot_list[index]
        
    def take_action(self, action, action_finished_callback=None):
        """通过动作名称或索引执行动作，并设置动作完成后的回调"""
        if isinstance(action, int):  # 如果传入的是索引，从热列表中取动作
            if 0 <= action < len(self.hot_list):
                action_name = self.hot_list[action]
                action_sequence = self.action_configs.get(action_name)
            else:
                log.debug(f"Invalid action index: {action}")
                return
        elif isinstance(action, str):  # 如果传入的是动作名称，直接取
            action_sequence = self.action_configs.get(action)
        else:
            log.debug(f"Invalid action type: {type(action)}")
            return

        if action_sequence:
            self.add_action(action_sequence, action_finished_callback=action_finished_callback)
        else:
            log.debug(f"Action not found: {action}")

    def is_running(self):
        """检查当前是否有动作在执行"""
        return not self.action_executed_event.is_set()

    def wait_for_finish(self, timeout=None):
        """
        阻塞等待直到当前动作执行完成或超时。
        timeout: 等待的最长时间（秒），为 None 时表示无限等待。
        """
        finished = self.action_executed_event.wait(timeout=timeout)
        if not finished:
            log.debug("等待动作完成超时")
    def is_interruptible(self, action_name):
        """
        判断给定动作是否可以被打断。
        
        参数:
            action_name (str): 动作名称。
        
        返回:
            bool: 如果动作可以被打断，返回 True；否则返回 False。
        """
        return action_name not in self.no_interrupts_set  
'''
# 外部接口示例
if __name__ == "__main__":
    # 初始化动作执行器
    executor = ActionExecutor('./config/actions_config.yaml')

    # 动作结束时的回调函数
    def on_action_finished():
        log.debug("动作执行完毕")

    # 示例：通过 DQN 网络获取 action_id 并执行对应动作，并注册动作完成回调
    action_id = 0  # 假设 DQN 网络输出的 action_id 是 0
    executor.take_action(action_id, action_finished_callback=on_action_finished)  # 通过索引执行动作

    # 轮询检查是否在执行中
    while executor.is_running():
        log.debug("动作正在执行中...")
        # 修改部分开始：精确控制打印间隔（可选）
        # time.sleep(0.5)  # 原来的睡眠方式
        time.sleep(0.5)  # 这里的睡眠不需要特别精确，因为只是用于打印状态
        # 修改部分结束

    # 中途打断动作（可选）
    # time.sleep(1)  # 假设 1 秒后需要打断动作
    # executor.interrupt_action()  # 调用打断函数
'''