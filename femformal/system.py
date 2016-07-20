import numpy as np

class System(object):

    def __init__(self, A, b, C=None):
        self.A = np.array(A)
        self.b = np.array(b)
        if C is not None:
            self.C = np.array(C)
        else:
            self.C = np.empty(shape=(0,0))

    def subsystem(self, indices):
        i = np.array(indices)
        A = self.A[i[:, None], i]
        b = self.b[i]

        j = np.setdiff1d(np.nonzero(self.A[i, :])[1], i)
        j = j[(j >= 0) & (j < self.n)]
        C = self.A[i[:,None], j]

        return System(A, b, C)

    @property
    def n(self):
        return len(self.A)

    @property
    def m(self):
        return self.C.shape[1]
