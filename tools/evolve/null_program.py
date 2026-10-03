"""Null candidate: measures the evolved harness's runtime memory floor.

It builds a one-page disk file and answers every query with ids 0..k-1, so the
search-phase peak anonymous memory of this program is the floor (Python, numpy,
faiss, numba, harness) that evolve_bench.py subtracts from every candidate.
"""

import numpy as np

SEARCH_POINTS = [{}]


def build(ctx):
    ctx.disk_writer("empty").write_pages(np.zeros((1, 4096), dtype=np.uint8))


class Searcher:
    def __init__(self, ctx, params):
        pass

    def search(self, query, k, io):
        return np.arange(k)
