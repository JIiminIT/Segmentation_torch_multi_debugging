import multiprocessing
import collections
from joblib import Parallel, delayed


__all__ = ["run_parallel"]


def run_parallel(task, items, n_jobs=multiprocessing.cpu_count(), check_pickle=True):
    """Parallelize tasks into multiple jobs with user custom parameters 

    Usage:
        .. code-block:: python

            import tinycat as cat

            def task(args):
                ...

            # use separate arguments and keyword arguments
            cat.multicore.run_parallel(task, args)

    Args:
        task (callable): task to be parallelized
        items (iterable): iterable object to feed as an argument
        n_jobs (int, optional): Defaults to multiprocessing.cpu_count(). Number of processes to be used
    
    Returns:
        Returned value of parallel task
    """

    assert callable(task)

    for i in range(len(items)):
        if not isinstance(items[i], collections.Iterable):
            items[i] = [items[i]]

    return Parallel(n_jobs=n_jobs)(
        delayed(task, check_pickle=check_pickle)(*arg) for arg in items
    )
