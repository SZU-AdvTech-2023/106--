from cmath import atan
from dis import dis
import numpy as np
import numpy.matlib as nma


def wthresh(X, SORH, T):
    """
    doing either hard (if SORH = 'h') or soft (if SORH = 's') thresholding

    Parameters
    ----------
    X: array
         input data (vector or matrix)
    SORH: str
         's': soft thresholding
         'h' : hard thresholding
    T: float
          threshold value

    Returns
    -------
    Y: array_like
         output

    Examples
    --------
    y = np.linspace(-1,1,100)
    thr = 0.4
    ythard = wthresh(y,'h',thr)
    ytsoft = wthresh(y,'s',thr)
    """
    if ((SORH) != 'h' and (SORH) != 's'):
        print(' SORH must be either h or s')

    elif (SORH == 'h'):
        Y = X * (np.abs(X) > T)
        return Y
    elif (SORH == 's'):
        res = (np.abs(X) - T)
        res = (res + np.abs(res)) / 2.
        Y = np.sign(X) * res
        return Y


def fista_fun2(L_At_A: np.ndarray, L_At_b: np.ndarray, lbd1, lbd2, max_iter,
               dim):
    t = 1
    tk = 1
    xk_1 = 0
    n = L_At_A.shape[0]
    yk = np.zeros((n, ))
    x = np.zeros_like(yk)

    for i in range(max_iter):
        y_Tem = yk - np.matmul(L_At_A, yk) + L_At_b
        # print('L_At_A:', L_At_A.shape)
        # print('yk:', yk.shape)
        # print('L_At_b:', L_At_b.shape)
        # print('y_Tem:', y_Tem.shape)
        x[:n - 3] = wthresh(y_Tem[:n - 3], 's', lbd1)
        x[n - 3:] = y_Tem[n - 3:] / tk / (1 / tk + lbd2)

        if dim == 1:
            x[n - 2] = (y_Tem[n - 2] / tk + lbd2) / (1 / tk + lbd2)
        elif dim == 2:
            x[n - 1] = (y_Tem[n - 1] / tk + lbd2) / (1 / tk + lbd2)

        tk = 1 / 2 + np.sqrt(1 + 4 * t * t) / 2
        AccWei = (t - 1) / tk
        # AccWei = 1
        t = tk
        yk = x + AccWei * (x - xk_1)
        xk_1 = x

    return x


def CoeffMultiC(P: np.ndarray, Q: np.ndarray, C: np.ndarray, lbd1: float,
                lbd2: float, max_iter: int):
    n = P.shape[0]

    # t = np.zeros((n, n))
    P1 = np.expand_dims(P, 0)
    P2 = np.expand_dims(P, 1)
    dist = np.linalg.norm(P1 - P2, axis=2)

    m = C.shape[1]

    dr = np.kron(dist, np.ones((1, m)))
    cr = nma.repmat(C, n, n)
    dr = dr / cr
    dr[dr > 1] = 1
    fista_v = np.power(1 - dr, 4) * (1 + 4 * dr)
    # print(dr)

    fista_A = np.zeros((n, n * m + 3))
    fista_A[:n, :m * n] = fista_v
    fista_A[:n, m * n] = 1
    fista_A[:n, m * n + 1] = P[:, 0]
    fista_A[:n, m * n + 2] = P[:, 1]

    At_A = np.matmul(fista_A.transpose(), fista_A)
    # print(np.linalg.eig(At_A))
    Lip = 1 / np.max(np.abs(np.linalg.eig(At_A)[0]))
    L_At_A = Lip * At_A

    At_bx = np.matmul(fista_A.transpose(), Q[:, 0])
    L_At_bx = Lip * At_bx

    At_by = np.matmul(fista_A.transpose(), Q[:, 1])
    L_At_by = Lip * At_by

    cx_FISTA = fista_fun2(L_At_A, L_At_bx, lbd1, lbd2, max_iter, 1)
    cy_FISTA = fista_fun2(L_At_A, L_At_by, lbd1, lbd2, max_iter, 2)

    c_FISTA = np.stack([cx_FISTA, cy_FISTA], 1)
    # print(c_FISTA.shape)

    return c_FISTA[:-3]


def move(points: np.ndarray, control_points: np.ndarray,
         coefficients: np.ndarray, c: np.ndarray):
    n_control_points = control_points.shape[0]
    m = c.shape[1]

    points = np.expand_dims(points, 1)
    control_points = np.expand_dims(control_points, 0)
    dist = np.linalg.norm(points - control_points, axis=2)
    dist = np.kron(dist, np.ones([1, m]))

    c = nma.repmat(c, 1, n_control_points)
    dist = dist / c
    dist[dist > 1] = 1
    fista_v = np.power(1 - dist, 4) * (1 + 4 * dist)
    # print(fista_v.shape)
    displacement = np.matmul(fista_v, coefficients)
    return displacement
