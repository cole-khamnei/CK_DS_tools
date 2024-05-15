import os
import time
import sys

import dill as pickle
import multiprocess as mp

from gc import get_referents
from thefuzz import fuzz

from typing import Any, Iterable, List, Optional, Tuple, Union
from types import ModuleType, FunctionType

# ------------------------------------------------------------------- #
# --------------------           Random          -------------------- #
# ------------------------------------------------------------------- #

class Timer:
    def __init__(self, print_on_exit: bool = True):
        self.start_time = time.time()
        self.final_time = False
        self.print_on_exit = print_on_exit

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.get_elapsed_time()
        self.final_time = True
        if self.print_on_exit:
            self.print_time()

    def get_elapsed_time(self):
        """"""
        if not self.final_time:
            self.elapsed_time = time.time() - self.start_time

    def print_time(self):
        """"""
        self.get_elapsed_time()
        print(f"Elapsed time: {self.elapsed_time:,.04f} seconds")


def fuzzy_index_search(term: str, descriptions: Iterable[str],
                       fuzzy_threshold: int = 95,
                       and_search: bool = False) -> List[int]:
    """ Searches a list of descriptions and returns any with a fuzzy index above a threshold."""

    if isinstance(term, str):
        term = term.lower()
        return [i for (i, desc) in enumerate(descriptions) if term in desc.lower()]
        # return [i for (i, desc) in enumerate(descriptions)
        #         if fuzz.partial_ratio(term, desc.lower()) > fuzzy_threshold]

    index_sets = [set(fuzzy_index_search(term_i, descriptions, fuzzy_threshold=fuzzy_threshold)) for term_i in term]

    final_indices = index_sets[0]
    for indices in index_sets[1:]:
        if and_search:
            final_indices = final_indices.intersection(indices)
        else:
            final_indices = final_indices.union(indices)

    return sorted(final_indices)


def exclude(main_list: List[Any], exclude_list: List[Any]) -> List[Any]:
    """ Excludes item found in another list"""
    return [item for item in main_list if item not in exclude_list]


"""
https://stackoverflow.com/questions/449560/how-do-i-determine-the-size-of-an-object-in-python
"""

# Custom objects know their class.
# Function objects seem to know way too much, including modules.
# Exclude modules as well.
BLACKLIST = type, ModuleType, FunctionType

def get_size(obj):
    """sum size of object & members."""

    if isinstance(obj, BLACKLIST):
        raise TypeError('get_size() does not take argument of type: ' + str(type(obj)))

    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


def tab_shift(text: str, tab_size: int = 4) -> str:
    tab = " " * tab_size
    return tab + text.replace("\n", f"\n{tab}").rstrip(tab)


def multiprocess_pool(function, param_sets: list, n_processes: int = 4):
    """ Runs a function in parallel using the pool multiprocess"""

    with mp.Pool(n_processes) as pool:
        results = [pool.apply_async(function, param_set) for param_set in param_sets]
        return [result.get() for result in results]


class MemoizedFunction:
    def __init__(self, function):
        self.function = function
        self.memory = {}

    def _call(self, *args):
        """"""
        key = str(args)
        value = self.memory.get(key, None)

        if not value:
            value = self.function(*args)
            self.memory[key] = value

        return value

    def __repr__(self) -> str:
        s = f"MemoizedFunction of {str(self.function).split(' of')[0]}>\n".replace(">>", ">")
        s += f"Current Memory: {get_size(self.memory)} Bytes"
        return s


def pickle_dump(obj, path: str):
    """"""
    with open(path, 'wb') as pickle_handle:
        pickle.dump(obj, pickle_handle)


def pickle_load(path: str):
    """"""
    with open(path, 'rb') as pickle_handle:
        return pickle.load(pickle_handle)


def cache(function, params, path: str, rewrite: bool = False):
    """"""
    if os.path.exists(path) and not rewrite:
        return pickle_load(path)

    obj = function(*params)

    pickle_dump(obj, path)
    return obj



# ------------------------------------------------------------------- #
# --------------------   Jupyter Notebook Tools  -------------------- #
# ------------------------------------------------------------------- #


def is_jupyter_notebook():
    """
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def tqdm_import():
    global tqdm
    if is_jupyter_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

# ------------------------------------------------------------------- #
# --------------------            End            -------------------- #
# ------------------------------------------------------------------- #

