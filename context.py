# context.py
import time
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import window
import grabscreen
from log import log


class Context:
    def __init__(self, frame_buffer_size=3):
        # 循环缓冲区大小
        self.frame_buffer_size = frame_buffer_size

        # 捕获一次画面 获取image的帧信息
        frame = grabscreen.grab_screen()
        window.BaseWindow.set_frame(frame)
        window.BaseWindow.update_all()

        # 获取当前帧的形状和状态信息
        frame_shape = window.battle_roi_window.color.shape
        frame_dtype = window.battle_roi_window.color.dtype
        frame_size  = window.battle_roi_window.color.nbytes
        status_dtype = [(k, 'f4') for k in self.get_all_status_keys()]
        status_size = np.dtype(status_dtype).itemsize

        # 计算共享内存所需大小，并创建共享内存
        # 前部分用于存储元数据，包括 frame_buffer_size，frame_shape, frame_dtype, status_dtype 等等
        metadata_size = (np.dtype('i').itemsize +  # 存储 frame_buffer_size
                         len(frame_shape) * np.dtype('i').itemsize +  # 存储 frame_shape
                         np.dtype('i').itemsize +  # 存储 frame_dtype.itemsize
                         np.dtype('i').itemsize)  # 存储 status_dtype.itemsize

        total_size = metadata_size + (frame_size + status_size) * frame_buffer_size
        self.shared_memory = shared_memory.SharedMemory(create=True, size=total_size)
        self.shared_memory_name = self.shared_memory.name

        # 存储元数据到共享内存开头部分
        self.store_metadata(metadata_size, frame_shape, frame_dtype, status_dtype)

        # 存储当前读写索引
        self.write_index = mp.Value('i', 0)
        self.read_index = mp.Value('i', 0)

        # 用于事件的两个队列
        self.emergency_event_queue = mp.Queue()
        self.normal_event_queue = mp.Queue()

        # 记录前一帧的状态信息
        self.previous_status = None

        # 时间戳
        self.last_timestamp = time.time()

        # 保存帧和状态的 dtype 和 shape
        self.frame_shape = frame_shape
        self.frame_dtype = frame_dtype
        self.frame_size = frame_size
        self.status_dtype = status_dtype
        self.metadata_size = metadata_size

    def reopen_shared_memory(self):
            """在子进程中重新打开共享内存"""
            self.shared_memory = shared_memory.SharedMemory(name=self.shared_memory_name)
            
    def store_metadata(self, metadata_size, frame_shape, frame_dtype, status_dtype):
        """将元数据存入共享内存"""
        metadata = np.ndarray(metadata_size // 4, dtype=np.int32, buffer=self.shared_memory.buf[:metadata_size])

        offset = 0
        metadata[offset] = self.frame_buffer_size
        offset += 1

        for i, dim in enumerate(frame_shape):
            metadata[offset + i] = dim
        offset += len(frame_shape)

        metadata[offset] = frame_dtype.itemsize
        offset += 1

        metadata[offset] = np.dtype(status_dtype).itemsize


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
        window.BaseWindow.set_frame(frame)
        window.BaseWindow.update_all()

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
            "q_found": window.q_window.check_similarity("./images/q.png", threshold=0.82)[0],
        }

        # 处理事件信息
        if self.previous_status:
            # 比较当前状态与前一状态，生成事件
            self.compare_status(current_status)

        # 更新前一帧状态信息
        self.previous_status = current_status.copy()

        # 将当前帧画面和状态信息写入共享内存
        self.write_frame_and_status(current_status)

    def compare_status(self, current_status):
        """比较当前状态和之前状态，生成事件"""
        # 仅对特定变量生成事件
        event_keys = self.get_all_status_keys()
        
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
                    if key == 'q_found':
                        log.debug(f"q_found,{event}")
                else:
                    self.normal_event_queue.put(event)  # 普通事件


    def write_frame_and_status(self, current_status):
        """将画面和状态信息写入共享内存"""
        # 获取当前画面
        frame = window.battle_roi_window.color

        # 写入共享内存中的索引
        index = self.write_index.value % self.frame_buffer_size

        # 计算帧和状态在共享内存中的偏移量
        buffer_offset = self.metadata_size + (self.frame_size + np.dtype(self.status_dtype).itemsize) * index

        # 写入画面
        np_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=self.shared_memory.buf[buffer_offset:buffer_offset + self.frame_size])
        np_frame[:] = frame

        # 写入状态信息
        status_offset = buffer_offset + self.frame_size
        np_status = np.ndarray(1, dtype=self.status_dtype, buffer=self.shared_memory.buf[status_offset:])
        for key in current_status:
            np_status[0][key] = current_status[key]

        # 更新写索引
        self.read_index.value = index 
        self.write_index.value = (index + 1) % self.frame_buffer_size



    def get_frame_and_status(self):
        """直接从共享内存和 read_index 读取 frame 和 status 信息"""
        
        # 从共享内存中提取元数据
        metadata = self.shared_memory.buf[:self.metadata_size]
        
        offset = 0
        frame_buffer_size = np.frombuffer(metadata, dtype=np.int32, count=1, offset=offset)[0]
        offset += np.dtype('i').itemsize
        
        frame_shape = tuple(np.frombuffer(metadata, dtype=np.int32, count=len(self.frame_shape), offset=offset))
        offset += len(self.frame_shape) * np.dtype('i').itemsize
        
        frame_dtype_size = np.frombuffer(metadata, dtype=np.int32, count=1, offset=offset)[0]
        offset += np.dtype('i').itemsize
        
        status_dtype_size = np.frombuffer(metadata, dtype=np.int32, count=1, offset=offset)[0]

        # 计算帧和状态数据大小
        frame_size = np.prod(frame_shape) * frame_dtype_size

        # 根据 read_index 计算当前要读取的帧和状态信息的偏移量
        index = self.read_index.value % frame_buffer_size
        buffer_offset = self.metadata_size + (frame_size + status_dtype_size) * index

        # 从共享内存中读取帧
        frame = np.ndarray(frame_shape, dtype=self.frame_dtype, buffer=self.shared_memory.buf[buffer_offset:buffer_offset + frame_size])

        # 读取状态信息
        status_offset = buffer_offset + frame_size
        status = np.ndarray(1, dtype=self.status_dtype, buffer=self.shared_memory.buf[status_offset:status_offset + status_dtype_size])

        return frame, status[0]

    def get_features(self,status):
        return [
            float(status['self_blood']) / 100.0,
            float(status['boss_blood']) / 100.0,
            float(status['self_energy']) / 100.0,
            float(status['self_magic']) / 100.0,
            float(status['hulu']) / 100.0,
            float(status['skill_1']),
            float(status['skill_2']),
            float(status['skill_3']),
            float(status['skill_4']),
            float(status['skill_ts']),
            float(status['skill_fb']),
            float(status['gunshi1']),
            float(status['gunshi2']),
            float(status['gunshi3']),
        ]

    def get_features_len(self):
        return 14

    def get_emergency_event_queue(self):
        return self.emergency_event_queue

    def get_normal_event_queue(self):
        return self.normal_event_queue
