"""
"""

from multiprocessing.synchronize import RLock as MultiprocessingRLock


try:
    from tqdm import tqdm as _tqdm
except ImportError: # pragma: no cover
    _tqdm = None


def remove_tqdm_multiprocessing_lock():
    if _tqdm is None: # pragma: no cover
        return
    tqdm_lock = _tqdm.get_lock()
    assert tqdm_lock.__class__.__name__ == 'TqdmDefaultWriteLock'
    tqdm_lock.locks = [
        lock for lock in tqdm_lock.locks
        if not isinstance(lock, MultiprocessingRLock)
    ]


tqdm = _tqdm
