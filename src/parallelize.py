from functools import wraps
from dask import delayed, compute
from dask.distributed import Client
import psutil

# A useful decorator to enable multiprocessing, based on dask.
def parallelize(cores: int = 0):
    if cores <= 0:
        cores = psutil.cpu_count(logical=False)

    def decorator(func):
        @wraps(func)
        def wrapper(tasks, *args, **kwargs):

            @delayed
            def _delayed_func(*args, **kwargs):
                return func(*args, **kwargs)
            
            all_tasks = [_delayed_func(task, *args, **kwargs) for task in tasks]
            with Client(n_workers=cores, threads_per_worker=1):
                results = compute(*all_tasks)

            return results
        return wrapper
    return decorator
