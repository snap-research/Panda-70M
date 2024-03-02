
from functools import wraps


class GlobalProfiler:
    """
    Helper class for defining a profiler that is accessible everywhere in the code
    """
    def __init__(self):
        self.profiler = None

    def set_profiler(self, profiler):
        self.profiler = profiler

    def record_function(self, name: str):
        """
        Records the function using the given name
        """
        return ProfilerContext(self.profiler, name)


class ProfilerContext:

    def __init__(self, profiler, name: str):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        if self.profiler is not None:
            self.profiler.start(self.name)
        return None

    def __exit__(self, type, value, traceback):
        if self.profiler is not None:
            self.profiler.stop(self.name)


profiler = GlobalProfiler()


def record_function(function):

    @wraps(function)
    def wrapper(*args, **kwargs):
        with profiler.record_function(function.__name__):
            return function(*args, **kwargs)
    return wrapper
