import psutil
import tracemalloc
import logging
import sys

import deepchem
from deepchem.feat.mol_graphs import ConvMol


log = logging.getLogger("ATOM")


def get_memory_usage() -> float:
    """Return the current memory usage of the Python process in GiB"""
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024 * 1024 * 1024)

    return rss_gb


def tracemalloc_snapshot():
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    log.debug("[ Top 10 ]")
    for stat in top_stats[:10]:
        log.debug(stat)


def sizeof_convmol(mol: ConvMol):
    assert type(mol) == ConvMol

    log.debug(f"Estimating size of ConvMol {mol}")

    # base size
    size = sys.getsizeof(mol)
    log.debug(f"Basic sizeof: {size}")

    # size of atom features (hopefully this is most of it)
    size += mol.atom_features.nbytes
    log.debug(f"Size of atom_features: {mol.atom_features.nbytes}")

    # these probably aren't going to be accurate
    log.debug(f"Size of deg_list: {sys.getsizeof(mol.deg_list)}")
    size += sys.getsizeof(mol.deg_list)
    log.debug(f"Size of deg_adj_lists: {sys.getsizeof(mol.deg_adj_lists)}")
    size += sys.getsizeof(mol.deg_adj_lists)
    log.debug(f"Size of canon_adj_list: {sys.getsizeof(mol.canon_adj_list)}")
    size += sys.getsizeof(mol.canon_adj_list)
    log.debug(f"Size of membership: {sys.getsizeof(mol.membership)}")
    size += sys.getsizeof(mol.membership)

    return size
