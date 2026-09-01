"""Optional resource telemetry. Only imported/used in the monitoring process."""
from __future__ import annotations

import os
import time

from .config import MonitoringConfig


class ResourceSampler:
    def __init__(self, parent_pid: int, config: MonitoringConfig) -> None:
        self.parent_pid = parent_pid
        self.config = config
        self.psutil = None
        self.nvml = None
        self.gpu = None
        self.errors: dict[str, str] = {}
        self.processes: dict[int, object] = {}
        self.io_previous: dict[int, tuple] = {}
        self.game_pids: set[int] = set()
        self.last_discovery = -float("inf")
        try:
            import psutil

            self.psutil = psutil
            psutil.cpu_percent(interval=None)  # prime; never report this first zero
            self._process(parent_pid)
            self._process(os.getpid())
        except ImportError:
            self.errors["cpu"] = "psutil not installed; install .[performance]"
        except Exception as error:
            self.errors["cpu_initialization"] = str(error)
        if config.gpu_enabled:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml = pynvml
                self.gpu = pynvml.nvmlDeviceGetHandleByIndex(config.gpu_index)
            except Exception as error:
                self.errors["gpu"] = f"NVML unavailable: {type(error).__name__}: {error}"
        else:
            self.errors["gpu"] = "disabled by configuration"

    def _process(self, pid: int):
        if pid not in self.processes:
            process = self.psutil.Process(pid)
            process.cpu_percent(interval=None)
            self.processes[pid] = process
        return self.processes[pid]

    def _sample_process(self, pid: int, now: float) -> dict:
        try:
            is_new = pid not in self.processes
            process = self._process(pid)
            with process.oneshot():
                cpu = process.cpu_percent(interval=None)
                result = {
                    "pid": pid,
                    "name": process.name(),
                    "cpu_percent_one_core": None if is_new else cpu,
                    "cpu_percent_machine": None if is_new else cpu / (self.psutil.cpu_count() or 1),
                    "rss_mb": process.memory_info().rss / 1024**2,
                    "threads": process.num_threads(),
                }
                try:
                    io = process.io_counters()
                    previous = self.io_previous.get(pid)
                    result["read_bytes_per_second"] = None
                    result["write_bytes_per_second"] = None
                    if previous and now > previous[0]:
                        result["read_bytes_per_second"] = max(0, io.read_bytes - previous[1]) / (now - previous[0])
                        result["write_bytes_per_second"] = max(0, io.write_bytes - previous[2]) / (now - previous[0])
                    self.io_previous[pid] = (now, io.read_bytes, io.write_bytes)
                except (AttributeError, self.psutil.Error):
                    result["io_unavailable"] = True
            return result
        except self.psutil.Error as error:
            self.processes.pop(pid, None)
            self.io_previous.pop(pid, None)
            return {"pid": pid, "unavailable": type(error).__name__}

    def sample(self) -> dict:
        now = time.perf_counter()
        result = {
            "timestamp": time.time(),
            "monotonic": now,
            "unavailable": dict(self.errors),
            "system": None,
            "recorder": None,
            "monitor": None,
            "game": [],
            "gpu": None,
        }
        if self.psutil is not None:
            if now - self.last_discovery >= 5.0:
                self.game_pids = {
                    process.info["pid"]
                    for process in self.psutil.process_iter(["pid", "name"])
                    if (process.info["name"] or "").lower() == self.config.game_process_name.lower()
                }
                self.last_discovery = now
            memory = self.psutil.virtual_memory()
            result["system"] = {
                "cpu_percent": self.psutil.cpu_percent(interval=None),
                "memory_percent": memory.percent,
                "available_memory_mb": memory.available / 1024**2,
            }
            result["recorder"] = self._sample_process(self.parent_pid, now)
            result["monitor"] = self._sample_process(os.getpid(), now)
            result["game"] = [self._sample_process(pid, now) for pid in sorted(self.game_pids)]
            if not self.game_pids:
                result["unavailable"]["game"] = f"process not found: {self.config.game_process_name}"
        if self.nvml is not None and self.gpu is not None:
            try:
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu)
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu)
                result["gpu"] = {
                    "index": self.config.gpu_index,
                    "scope": "whole_device_not_game_or_torch_only",
                    "utilization_percent": utilization.gpu,
                    "memory_controller_percent": utilization.memory,
                    "memory_used_mb": memory.used / 1024**2,
                    "memory_total_mb": memory.total / 1024**2,
                }
                for name, read in (
                    ("temperature_c", lambda: self.nvml.nvmlDeviceGetTemperature(self.gpu, self.nvml.NVML_TEMPERATURE_GPU)),
                    ("power_w", lambda: self.nvml.nvmlDeviceGetPowerUsage(self.gpu) / 1000),
                ):
                    try:
                        result["gpu"][name] = read()
                    except Exception:
                        result["gpu"][name] = None
            except Exception as error:
                result["unavailable"]["gpu"] = str(error)
        return result

    def close(self) -> None:
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
