import threading
import time

lock = threading.Lock()


# --- 不加锁的情况 ---
shared_resource = 0

def increment():
    global shared_resource
    # 模拟非原子操作：
    # 1. 读取当前值
    temp = shared_resource
    # 2. 模拟一些耗时操作，增加线程切换的可能性
    time.sleep(0.001)
    # 3. 写回新值
    shared_resource = temp + 1

threads = []
for _ in range(100):
    t = threading.Thread(target=increment)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"不加锁的结果: {shared_resource}") # 结果通常会小于 100


# --- 加锁的情况 --- 
shared_resource = 0 # 重置资源

def increment_with_lock():
    global shared_resource
    with lock:  # 自动获取和释放锁
        temp = shared_resource
        time.sleep(0.001)
        shared_resource = temp + 1

threads = []
for _ in range(100):
    t = threading.Thread(target=increment_with_lock)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"加锁后的结果: {shared_resource}") # 结果总是 100