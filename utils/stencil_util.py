import numpy as np

class stencilOrder:
    def __init__(self, order) -> None:
        assert type(order) == list or type(order) == tuple or type(order) == np.ndarray
        self.order = np.array(order)
        assert self.order.shape[1] == 3
        self._reverse_map()

    def __len__(self):
        return self.order.shape[0]

    def _reverse_map(self):
        self.reverse_map = {}
        for i, dir in enumerate(self.order):
            self.reverse_map[tuple(dir)] = i

    def dir2idx(self, dir):
        dir = np.array(dir)
        assert dir.shape == (3,)
        dir = np.round(dir) if dir.dtype != np.int else dir
        return self.reverse_map[tuple(dir.astype(int))]

stencil_6 = stencilOrder([(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)])
stencil_26 = stencilOrder([(i,j,k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1] if (i,j,k) != (0,0,0)])

def get_stencil(n_neighbor):
    assert n_neighbor in [6, 26]
    return globals()[f'stencil_{n_neighbor}']