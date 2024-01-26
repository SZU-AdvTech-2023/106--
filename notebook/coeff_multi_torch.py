import torch
import numpy as np
import torch.nn.functional as F


def wthresh(X: torch.Tensor, SORH, T):
    if ((SORH) != 'h' and (SORH) != 's'):
        print(' SORH must be either h or s')
    elif (SORH == 'h'):
        Y = X * (torch.abs(X) > T)
        return Y
    elif (SORH == 's'):
        res = (torch.abs(X) - T)
        res = (res + torch.abs(res)) / 2.
        Y = torch.sign(X) * res
        return Y


def wthreshbyblock(X: torch.Tensor, group_size):
    batch_size = X.size()[0]
    cp_num = X.size()[1] // group_size
    X_cp = X.view([batch_size, cp_num, group_size])
    X_cp_max = torch.max(torch.abs(X_cp), dim=2, keepdim=True)
    # print(X_cp_max.size())
    X_cp_filtered = X_cp * (torch.abs(X_cp) == X_cp_max)
    return X_cp_filtered.view([batch_size, -1, 1])


def fista_fun2(L_At_A: torch.Tensor, L_At_b: torch.Tensor, lbd1, lbd2,
               max_iter, dim):
    t = 1
    tk = 1
    xk_1 = 0
    batch = L_At_A.size()[0]
    n = L_At_A.size()[1]
    yk = torch.zeros((batch, n, 1)).cuda()
    x = torch.zeros_like(yk).cuda()

    for i in range(max_iter):
        y_Tem = yk - torch.bmm(L_At_A, yk) + L_At_b

        # x[:, :n - 3] = wthreshbyblock(y_Tem[:, :n - 3], 3)
        # x[:, :n - 3] = wthresh(x[:, :n - 3], 's', lbd1)
        x[:, :n - 3] = wthresh(y_Tem[:, :n - 3], 's', lbd1)
        x[:, n - 3:] = y_Tem[:, n - 3:] / (tk / (1 / tk + lbd2))

        if dim == 1:
            x[:, n - 2] = (y_Tem[:, n - 2] / tk + lbd2) / (1 / tk + lbd2)
        elif dim == 2:
            x[:, n - 1] = (y_Tem[:, n - 1] / tk + lbd2) / (1 / tk + lbd2)

        tk = 1 / 2 + np.sqrt(1 + 4 * t * t) / 2

        AccWei = (t - 1) / tk
        t = tk
        yk = x + AccWei * (x - xk_1)
        xk_1 = x
    return x


def CoeffMultiC(P: torch.Tensor, Q: torch.Tensor, C: torch.Tensor, lbd1: float,
                lbd2: float, max_iter: int):

    n = P.size()[1]

    P1 = P.unsqueeze(1)
    P2 = P.unsqueeze(2)
    dist = torch.norm(P1 - P2, dim=3)

    m = C.size()[2]

    batch = P.size()[0]
    dr_dist = []
    for i in range(batch):
        dr_dist.append(torch.kron(dist[i], torch.ones((1, m)).cuda()))
    dr = torch.stack(dr_dist, 0)
    cr = C.unsqueeze(2).repeat([1, 1, n, 1]).view([1, 1, -1])
    dr = dr / cr
    dr_mask = dr <= 1
    fista_v = torch.pow(1 - dr, 4) * (1 + 4 * dr) * dr_mask

    fista_A = torch.zeros((batch, n, n * m + 3)).cuda()
    fista_A[:, :n, :m * n] = fista_v
    fista_A[:, :n, m * n] = 1
    fista_A[:, :n, m * n + 1] = P[:, :, 0]
    fista_A[:, :n, m * n + 2] = P[:, :, 1]

    At_A = torch.bmm(fista_A.permute(0, 2, 1), fista_A)
    Lip = []
    for i in range(batch):
        # print(torch.abs(torch.linalg.eig(At_A[i, :, :])[0]))
        Lip.append(1 / torch.max(torch.abs(torch.linalg.eig(At_A[i])[0])))
    Lip = torch.tensor(Lip).cuda().view([batch, 1, 1])
    L_At_A = Lip * At_A

    At_bx = torch.bmm(fista_A.permute(0, 2, 1), Q[:, :, 0].unsqueeze(2))
    L_At_bx = Lip * At_bx

    At_by = torch.bmm(fista_A.permute(0, 2, 1), Q[:, :, 1].unsqueeze(2))
    L_At_by = Lip * At_by

    cx_FISTA = fista_fun2(L_At_A, L_At_bx, lbd1, lbd2, max_iter, 1)
    cy_FISTA = fista_fun2(L_At_A, L_At_by, lbd1, lbd2, max_iter, 2)

    c_FISTA = torch.concat([cx_FISTA, cy_FISTA], 2)

    return c_FISTA[:, :-3]


def move(points: torch.Tensor, control_points: torch.Tensor,
         coefficients: torch.Tensor, c: torch.Tensor):
    n_control_points = control_points.size()[1]
    m = c.size()[2]

    batch = points.size()[0]

    points = points.unsqueeze(2)
    control_points = control_points.unsqueeze(1)
    dist = torch.norm(control_points - points, dim=3)

    dist_list = []
    for i in range(batch):
        dist_list.append(torch.kron(dist[i], torch.ones((1, m)).cuda()))
    dist = torch.stack(dist_list, 0)

    cr = c.unsqueeze(2).repeat([1, 1, n_control_points, 1]).view([1, 1, -1])
    # print(dist.size(), cr.size())
    dist = dist / cr
    dist_mask = dist <= 1
    fista_v = torch.pow(1 - dist, 4) * (1 + 4 * dist) * dist_mask

    displacement = torch.bmm(fista_v, coefficients)
    # print(displacement.size())
    return displacement
