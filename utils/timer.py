import time
from contextlib import contextmanager
from threading import Lock


class Timer:
    def __init__(self):
        self.__global_lock = Lock()
        self.__locks = {}
        self.__durations = {}

    def __get_new_lock(self, key):
        with self.__global_lock:
            _lock = self.__locks.get(key, None) or Lock()
            self.__locks[key] = _lock
        return _lock

    def __get_duration(self, key):
        with self.__global_lock:
            _duration = self.__durations.get(key)
        return _duration

    def __set_duration(self, key, duration):
        with self.__global_lock:
            self.__durations[key] = duration

    def __getitem__(self, key):
        return self.__get_duration(key)

    @contextmanager
    def with_time(self, key):
        with self.__get_new_lock(key):
            _start_time = time.time()
            yield
            _end_time = time.time()

        _duration = _end_time - _start_time
        self.__set_duration(key, _duration)

        return _duration


if __name__ == "__main__":
    timer = Timer()
    with timer.with_time("time_1"):
        time.sleep(2)

    with timer.with_time("time_2"):
        time.sleep(2.8)

    print(timer["time_1"], timer["time_2"])
