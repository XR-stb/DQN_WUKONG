import time
import multiprocessing as mp
import numpy as np
import window
import grabscreen

class Context:
    def __init__(self, frame_buffer_size=3):
        # 循环缓冲区大小
        self.frame_buffer_size = frame_buffer_size

        # 获取当前帧的形状和状态信息
        frame_shape = window.game_window.color.shape
        frame_dtype = window.game_window.color.dtype
        status_dtype = [(k, 'f4') for k in self.get_all_status_keys()]

        # 创建用于存放画面和状态信息的共享内存
        self.shared_frame_memory = mp.shared_memory.SharedMemory(create=True, size=np.prod(frame_shape) * frame_dtype.itemsize * self.frame_buffer_size)
        self.shared_status_memory = mp.shared_memory.SharedMemory(create=True, size=np.dtype(status_dtype).itemsize * self.frame_buffer_size)

        # 存储当前读写索引
        self.write_index = mp.Value('i', 0)
        self.read_index = mp.Value('i', 0)

        # 用于事件的两个队列
        self.emergency_event_queue = mp.SimpleQueue()
        self.normal_event_queue = mp.SimpleQueue()

        # 记录前一帧的状态信息
        self.previous_status = None

        # 时间戳
        self.last_timestamp = time.time()

        # 保存帧和状态的 dtype 和 shape
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype
        self.status_dtype = status_dtype

    def get_all_status_keys(self):
        """返回所有状态变量的键名"""
        return [
            "self_blood", "self_magic", "self_energy", "hulu", "boss_blood",
            "skill_1", "skill_2", "skill_3", "skill_4",
            "skill_ts", "skill_fb", "gunshi1", "gunshi2", "gunshi3", "q_found"
        ]

    def update_status(self):

        # 捕获最新的画面
        frame = grabscreen.grab_screen()
        BaseWindow.set_frame(frame)
        BaseWindow.update_all()

        # 获取当前状态
        current_status = {
            "self_blood": window.self_blood_window.get_status(),
            "self_magic": window.self_magic_window.get_status(),
            "self_energy": window.self_energy_window.get_status(),
            "hulu": window.hulu_window.get_status(),
            "boss_blood": window.boss_blood_window.get_status(),
            "skill_1": window.skill_1_window.get_status(),
            "skill_2": window.skill_2_window.get_status(),
            "skill_3": window.skill_3_window.get_status(),
            "skill_4": window.skill_4_window.get_status(),
            "skill_ts": window.skill_ts_window.get_status(),
            "skill_fb": window.skill_fb_window.get_status(),
            "gunshi1": window.gunshi1_window.get_status(),
            "gunshi2": window.gunshi2_window.get_status(),
            "gunshi3": window.gunshi3_window.get_status(),
            "q_found": window.q_window.check_similarity("./images/q.png", threshold=0.9)[0],
        }

        # 处理事件信息
        if self.previous_status:
            # 比较当前状态与前一状态，生成事件
            self.compare_status(current_status)

        # 更新前一帧状态信息
        self.previous_status = current_status

        # 将当前帧画面和状态信息写入共享内存
        self.write_frame_and_status(current_status)

    def compare_status(self, current_status):
        """比较当前状态和之前状态，生成事件"""
        # 仅对特定变量生成事件
        event_keys = ["self_blood", "boss_blood", "self_energy", "q_found"]
        
        for key in event_keys:
            if current_status[key] != self.previous_status[key]:
                relative_change = current_status[key] - self.previous_status[key]
                event = {
                    'event': key,
                    'relative_change': relative_change,
                    'previous_value': self.previous_status[key],
                    'current_value': current_status[key],
                    'timestamp': time.time()
                }
                if key in ["self_blood", "q_found"]:
                    self.emergency_event_queue.put(event)  # 紧急事件
                else:
                    self.normal_event_queue.put(event)  # 普通事件

    def write_frame_and_status(self, current_status):
        """将画面和状态信息写入共享内存"""
        # 获取当前画面
        frame = window.game_window.color

        # 写入共享内存中的索引
        index = self.write_index.value % self.frame_buffer_size

        # 写入画面
        frame_size = frame.nbytes
        frame_buffer_offset = frame_size * index
        np_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=self.shared_frame_memory.buf[frame_buffer_offset:frame_buffer_offset + frame_size])
        np_frame[:] = frame

        # 写入状态信息
        status_buffer_offset = np.dtype(self.status_dtype).itemsize * index
        np_status = np.ndarray(1, dtype=self.status_dtype, buffer=self.shared_status_memory.buf[status_buffer_offset:])
        for key in current_status:
            np_status[0][key] = current_status[key]

        # 更新写索引
        with self.write_index.get_lock():
            self.write_index.value = (self.write_index.value + 1) % self.frame_buffer_size

    def get_latest_frame_and_status(self):
        """返回最新的帧画面和状态信息"""
        index = self.read_index.value % self.frame_buffer_size

        # 读取画面
        frame_size = window.game_window.color.nbytes
        frame_buffer_offset = frame_size * index
        frame = np.ndarray(self.frame_shape, dtype=self.frame_dtype, buffer=self.shared_frame_memory.buf[frame_buffer_offset:frame_buffer_offset + frame_size])

        # 读取状态信息
        status_buffer_offset = np.dtype(self.status_dtype).itemsize * index
        status = np.ndarray(1, dtype=self.status_dtype, buffer=self.shared_status_memory.buf[status_buffer_offset:])

        return frame, status[0]

    @staticmethod
    def get_features(status_snapshot):
        """静态方法：获取归一化后的特征向量"""
        return [
            float(status_snapshot['self_blood']) / 100.0,
            float(status_snapshot['boss_blood']) / 100.0,
            float(status_snapshot['self_energy']) / 100.0,
            float(status_snapshot['self_magic']) / 100.0,
            float(status_snapshot['q_found'])  # 布尔值保持为 0 或 1
        ]

    def get_emergency_event_queue(self):
        return self.emergency_event_queue

    def get_normal_event_queue(self):
        return self.normal_event_queue