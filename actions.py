# coding=utf-8
import keys
from getkeys import key_check
import window
import restart

import time
import threading
from context import Context
from log import log

keys = keys.Keys()

# 创建一个动作执行器类，用于管理动作的异步执行
class ActionExecutor:
    def __init__(self):
        # 用于控制动作执行线程的停止
        self.running = True
        # 动作队列
        self.action_queue = []
        # 创建一个线程锁，确保线程安全
        self.lock = threading.Lock()
        # 启动动作执行线程
        self.thread = threading.Thread(target=self._execute_actions)
        self.thread.daemon = True  # 设置为守护线程
        self.thread.start()

    def add_action(self, action_func, *args, delay=0, **kwargs):
        # 将动作及其执行时间添加到队列
        execute_time = time.time() + delay
        with self.lock:
            self.action_queue.append((execute_time, action_func, args, kwargs))

    def _execute_actions(self):
        while self.running:
            current_time = time.time()
            with self.lock:
                queue_copy = self.action_queue.copy()
            for item in queue_copy:
                execute_time, action_func, args, kwargs = item
                if current_time >= execute_time:
                    try:
                        action_func(*args, **kwargs)
                    except Exception as e:
                        log(f"动作执行异常：{e}")
                    with self.lock:
                        if item in self.action_queue:
                            self.action_queue.remove(item)
            # 防止CPU占用过高
            time.sleep(0.001)

    def stop(self):
        self.running = False
        self.thread.join()

# 实例化动作执行器
action_executor = ActionExecutor()

def precise_sleep(target_duration):
    """精确睡眠函数，用于阻塞指定的时间。"""
    start_time = time.perf_counter()
    while True:
        current_time = time.perf_counter()
        elapsed_time = current_time - start_time
        if elapsed_time >= target_duration:
            break
        time.sleep(max(0, target_duration - elapsed_time))

def precise_sleep_non_blocking(target_duration):
    # 这个函数在当前实现中不需要执行任何操作
    pass

def pause(second=0.04):
    action_executor.add_action(keys.directKey, "T")
    action_executor.add_action(keys.directKey, "T", keys.key_release, delay=second)

def mouse_move(times):
    for i in range(times):
        action_executor.add_action(keys.directMouse, i, 0)
        # 添加延迟
        time_per_move = 0.004
        action_executor.add_action(lambda: None, delay=time_per_move)

def mouse_move_v2(times):
    for i in range(times):
        action_executor.add_action(keys.directMouse, i, 0)
        # 添加延迟
        time_per_move = 0.01
        action_executor.add_action(lambda: None, delay=time_per_move)

def lock_view(second=0.2):
    action_executor.add_action(keys.directMouse, buttons=keys.mouse_mb_press)
    action_executor.add_action(keys.directMouse, buttons=keys.mouse_mb_release, delay=second)

def run_with_direct(second=0.5, direct="W"):
    action_executor.add_action(keys.directKey, direct)
    action_executor.add_action(keys.directKey, "LSHIFT")
    action_executor.add_action(keys.directKey, direct, keys.key_release, delay=second)
    action_executor.add_action(keys.directKey, "LSHIFT", keys.key_release, delay=second)

def eat(second=0.04):
    action_executor.add_action(keys.directKey, "R")
    action_executor.add_action(keys.directKey, "R", keys.key_release, delay=second)

def recover():
    log("打药")
    #run_with_direct(2, "S")
    eat()
    

def attack(second=0.2):
    action_executor.add_action(keys.directMouse, buttons=keys.mouse_lb_press)
    action_executor.add_action(keys.directMouse, buttons=keys.mouse_lb_release, delay=second)
    log("攻击")

def heavy_attack(second=0.2):
    action_executor.add_action(keys.directMouse, buttons=keys.mouse_rb_press)
    action_executor.add_action(keys.directMouse, buttons=keys.mouse_rb_release, delay=second)
    log("重击")


def jump(second=0.04):
    action_executor.add_action(keys.directKey, "LCTRL")
    action_executor.add_action(keys.directKey, "LCTRL", keys.key_release, delay=second)

# 闪避
def dodge(second=0.04):
    action_executor.add_action(keys.directKey, "SPACE")
    action_executor.add_action(keys.directKey, "SPACE", keys.key_release, delay=second)
    log("闪避")

def go_forward(second=0.04):
    action_executor.add_action(keys.directKey, "W")
    action_executor.add_action(keys.directKey, "W", keys.key_release, delay=second)
    log("向前")

def go_back(second=0.04):
    action_executor.add_action(keys.directKey, "S")
    action_executor.add_action(keys.directKey, "S", keys.key_release, delay=second)

def go_left(second=0.04):
    action_executor.add_action(keys.directKey, "A")
    action_executor.add_action(keys.directKey, "A", keys.key_release, delay=second)

def go_right(second=0.04):
    log("向右ing")
    action_executor.add_action(keys.directKey, "D")
    action_executor.add_action(keys.directKey, "D", keys.key_release, delay=second)

def press_E(second=0.04):
    action_executor.add_action(keys.directKey, "E")
    action_executor.add_action(keys.directKey, "E", keys.key_release, delay=second)

def press_ESC(second=0.04):
    action_executor.add_action(keys.directKey, "ESC")
    action_executor.add_action(keys.directKey, "ESC", keys.key_release, delay=second)

def press_Enter(second=0.04):
    action_executor.add_action(keys.directKey, "ENTER")
    action_executor.add_action(keys.directKey, "ENTER", keys.key_release, delay=second)
    log("按下 Enter")

def press_up(second=0.04):
    action_executor.add_action(keys.directKey, "UP")
    action_executor.add_action(keys.directKey, "UP", keys.key_release, delay=second)
    log("按下 上箭头")

def press_down(second=0.04):
    action_executor.add_action(keys.directKey, "DOWN")
    action_executor.add_action(keys.directKey, "DOWN", keys.key_release, delay=second)
    log("按下 下箭头")

def press_left(second=0.04):
    action_executor.add_action(keys.directKey, "LEFT")
    action_executor.add_action(keys.directKey, "LEFT", keys.key_release, delay=second)
    log("按下 左箭头")

def press_right(second=0.04):
    action_executor.add_action(keys.directKey, "RIGHT")
    action_executor.add_action(keys.directKey, "RIGHT", keys.key_release, delay=second)
    log("按下 右箭头")

def use_skill(skill_key="1", second=0.04):
    action_executor.add_action(keys.directKey, skill_key)
    action_executor.add_action(keys.directKey, skill_key, keys.key_release, delay=second)
    log("使用 技能{}".format(skill_key))

def pause_game(context: Context, emergence_break=0):
    pressed_keys = key_check()
    
    if "T" in pressed_keys:
        if context.paused:
            context.paused = False
            log("start game")
        else:
            context.paused = True
            log("pause game, press T to start")

    if context.paused:
        log("paused")
        time.sleep(0.5) # 防止CPU占用过高 连按检测
        if emergence_break == 100:
            restart.restart()

        while True:
            pressed_keys = key_check()
            if "T" in pressed_keys:
                if context.paused:
                    context.paused = False
                    context.begin_time = int(time.time())
                    log("start game")
                    break
            time.sleep(0.01)  # 防止CPU占用过高
        return context
    return context

def is_skill_1_in_cd():
    return window.get_skill_1_window().blood_count() < 50

def take_action(action, context: Context) -> Context:
    # 更新当前状态
    context.magic_num = window.get_self_magic_window().blood_count()
    context.self_energy = window.get_self_energy_window().blood_count()
    context.self_blood = window.get_self_blood_window().blood_count()

    if action == 0:  # 无操作，等待
        #run_with_direct(0.5, 'W')
        context.dodge_weight = 1

    elif action == 1:  # 攻击
        attack()
        context.dodge_weight = 1

    elif action == 2:  # 闪避
        if context.magic_num > 10:
            dodge()
            context.dodge_weight += 10  # 防止收敛于一直闪避的状态

    elif action == 3:  # 使用技能1
        log(
            "magic:%d, energy:%d, self_blood:%d" % (context.magic_num, context.self_energy, context.self_blood)
        )
        log("skill_1 cd :%d" % is_skill_1_in_cd())
        if context.magic_num > 20 and not is_skill_1_in_cd():
            use_skill("1")
            attack()
            context.dodge_weight = 1

    elif action == 4:  # 恢复
        if context.self_blood < 80 and context.init_medicine_nums > 0:
            # 回体力和打药
            recover()
            context.init_medicine_nums -= 1
            context.dodge_weight = 1


    elif action == 5:  # 重击
        heavy_attack()
        context.dodge_weight = 1

    return context

# 在程序结束时，确保停止动作执行器线程
def stop_action_executor():
    action_executor.stop()
