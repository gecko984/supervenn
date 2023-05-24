import numpy as np
cimport cython

DIST_MAX = 10**9


def solve_tsp(arr, row_weights=None):
    n = arr.shape[1]
    if n <= 2:
        return [0, 1]
    arr_T = np.array(arr.T, dtype=np.int32)
    graph = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            graph[i, j] = np.sum(np.bitwise_xor(arr_T[i], arr_T[j]))
    path = list(faster_multichristofides_cython(graph)[0])

    from supervenn._algorithms import get_total_gaps_in_rows

    best_path = path[:]
    best_total_gaps = get_total_gaps_in_rows(arr[:, path], row_weights=row_weights)
    for i in range(n-1):
        path.append(path.pop(0))
        total_gaps = get_total_gaps_in_rows(arr[:, path], row_weights=row_weights)
        if best_total_gaps > total_gaps:
            best_total_gaps = total_gaps
            best_path = path[:]
    return best_path


@cython.boundscheck(False)
@cython.wraparound(False)
def two_opt(graph: np.array, path: list, int length, int two_opt_iters_max):
    cdef int[:, :] graph_view = graph
    path_np = np.array(path, dtype=np.int32)
    cdef int[:] path_view = path_np
    cdef int _, i, j, length_prev, delta, n = graph.shape[0]
    for _ in range(two_opt_iters_max):
        length_prev = length
        for i in range(1, n-3):
            for j in range(i+2, n-1):
                delta = graph_view[path_view[i-1], path_view[j-1]] + \
                    graph_view[path_view[i], path_view[j]] - \
                    graph_view[path_view[i-1], path_view[i]] - \
                    graph_view[path_view[j-1], path_view[j]]
                if delta < 0:
                    length += delta
                    path_view[i:j] = path_view[j-1:i-1:-1]
        if length == length_prev:
            break
    return list(path_np), int(length)


def faster_christofides_cython(graph: np.array, two_opt_iters_max=10):
    '''
    Faster christofides (with approximate minimum-weight perfect matching)

    Complexity: (without 2-opt https://en.wikipedia.org/wiki/2-opt) - O(n^2),
    single 2-opt iteration - O(n^2) ... O(n^3);
    '''
    christ = Christofides(graph)
    path, length = christ.find_path()
    return two_opt(graph, path, length, two_opt_iters_max)


def faster_multichristofides_cython(graph: np.array, two_opt_iters_max=10):
    '''
    Faster christofides (with approximate minimum-weight perfect matching)
    which runs n times and chooses the shortest path found

    Complexity: (without 2-opt https://en.wikipedia.org/wiki/2-opt) - O(n^2),
    single 2-opt iteration - O(n^2) ... O(n^3);
    '''
    path_best, length_best = None, DIST_MAX
    christ = Christofides(graph)
    for start_pos in range(graph.shape[0]):
        path, length = christ.find_path(start_pos)
        path, length = two_opt(graph, path, length, two_opt_iters_max)
        if length_best > length:
            length_best = length
            path_best = path[:]
    assert path_best
    return path_best, length_best


class Christofides:
    def __init__(self, graph: np.array):
        self.n = graph.shape[0]
        self.odds = []
        self.adjlist = [[] for _ in range(self.n)]
        self.graph = np.array(graph, dtype=np.int32)

        self.find_mst()
        self.perfect_matching()

    def find_mst(self):
        # prim's algorithm
        cdef int n = self.n
        cdef int[:, :] graph_view = self.graph
        key = np.full(n, DIST_MAX, dtype=np.int32)
        cdef int[:] key_view = key
        parent = np.zeros(n, dtype=np.int32)
        cdef int[:] parent_view = parent
        in_mst = np.zeros(n, dtype=np.uint8)
        cdef unsigned char[:] in_mst_view = in_mst
        key[0] = 0
        parent[0] = -1
        cdef int i, j, key_min, v, u
        for i in range(n-1):
            key_min, v = DIST_MAX, 0
            for j in range(n):
                if not in_mst_view[j] and key_view[j] < key_min:
                    key_min, v = key_view[j], j
            in_mst_view[v] = 1 # True
            for u in range(n):
                if graph_view[v, u] and in_mst_view[u] == 0 and graph_view[v, u] < key_view[u]:
                    parent_view[u] = v
                    key_view[u] = graph_view[v, u]
        for v1 in range(n):
            v2 = parent[v1]
            if v2 != -1:
                self.adjlist[v1].append(v2)
                self.adjlist[v2].append(v1)

    def perfect_matching(self):
        for r in range(self.n):
            if len(self.adjlist[r]) % 2 == 1:
                self.odds.append(r)
        closest = DIST_MAX
        while self.odds:
            length, first = DIST_MAX, self.odds.pop()
            for v in self.odds:
                if length > self.graph[first, v]:
                    length = self.graph[first, v]
                    closest = v

            self.adjlist[first].append(closest)
            self.adjlist[closest].append(first)
            self.odds.remove(closest)

    def find_euler_cycle(self, pos: int):
        adjlist = [x[:] for x in self.adjlist]
        path, stk = [], []
        stk = []
        while stk or adjlist[pos]:
            if adjlist[pos]:
                stk.append(pos)
                neighbor = adjlist[pos].pop()
                adjlist[neighbor].remove(pos)
                pos = neighbor
            else:
                path.append(pos)
                pos = stk.pop()

        path.append(pos)
        return path

    def make_hamilton_cycle(self, path: list):
        assert path
        length = 0
        curr, next = 0, 1
        visited = [False] * self.n
        root = path[0]
        visited[root] = True
        while next != len(path):
            if visited[path[next]]:
                path.pop(next)
            else:
                length += self.graph[path[curr], path[next]]
                visited[path[next]] = True
                curr, next = next, next + 1

        length += self.graph[path[curr], root]
        return path, length

    def find_path(self, start_pos=0):
        path = self.find_euler_cycle(start_pos)
        return self.make_hamilton_cycle(path)
