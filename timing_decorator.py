import time
from log import log
from functools import wraps

def timeit(func):
    """
    装饰器，用于测量函数的执行时间并打印出来。
    
    使用示例：
    
    @timeit
    def my_function():
        pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)
        end_time = time.time()    # 记录结束时间
        elapsed_time = end_time - start_time
        log.debug(f"{func.__name__} 耗时: {elapsed_time:.6f} 秒")  # 打印耗时
        return result
    return wrapper
