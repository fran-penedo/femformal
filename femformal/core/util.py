"""List and string utilities"""
from __future__ import division, absolute_import, print_function

import logging
import itertools

import numpy as np


logger = logging.getLogger(__name__)

def label(name, i, j):
    """Makes a name_i_j label

    Parameters
    ----------
    name : str
    i, j : int

    Returns
    -------
    str
        name_i_j

    """
    return name + "_" + str(i) + "_" + str(j)

def unlabel(label):
    """Splits a name_i_j label into (name, i, j)

    Parameters
    ----------
    label : str

    Returns
    -------
    name : str
    i, j : int

    """
    sp = label.split("_")
    return sp[0], int(sp[1]), int(sp[2])


def state_label(l):
    """Makes a label from a state list

    Parameters
    ----------
    l : iterable

    Returns
    -------
    str
        A label of the form s_ijk...

    """
    return 's' + '_'.join([str(x) for x in l])

def label_state(l):
    """Obtains a state list [i, j, k] from a s_ijk label

    Parameters
    ----------
    l : str
        A label of the form s_ijk

    Returns
    -------
    list
        The corresponding state list ([i,j,k])

    """
    return [int(x) for x in l[1:].split('_')]


def first_change(a, b):
    try:
        return next(((i, x[1] - x[0]) for i, x in enumerate(zip(a, b)) if x[0] != x[1]))
    except StopIteration:
        return 0, 0


def make_groups(l, n):
    """Group a list in groups of a given length

    If lover = len(l) % n > 0, make the first lover groups bigger

    Parameters
    ----------
    l : list
    n : int

    Returns
    -------
    list of lists

    """
    leftover = len(l) % n
    at = leftover * (n + 1)
    if leftover > 0:
        bigger = make_groups(l[:at], n+1)
    else:
        bigger = []
    rest = l[at:]
    return bigger + [rest[n * i: n * (i + 1)] for i in range(len(rest) // n)]


def project_states(states):
    x = np.array(states)
    return [list(set(c)) for c in x.T]


def project_list(l, indices):
    return [l[i] for i in indices]

def list_extr_points(l):
    return [[x[0], x[-1]] for x in l]

def project_regions(regions, indices):
    ret = {}
    for key, value in regions.items():
        ret[key] = project_list(value, indices)
    return ret


def cycle(n):
    """Iterate over all cycles of length n

    Parameters
    ----------
    n : int

    Yields
    ------
    list

    """
    cyc = itertools.cycle(range(n))
    for i in range(n):
        yield itertools.islice(cyc, n)
        cyc.next()
