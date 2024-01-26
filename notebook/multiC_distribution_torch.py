# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors

from coeff_multi_torch import CoeffMultiC, move
from load_york import get_orignal_data, check_accept

# %%
training_pair_path = 'D:\Code\cardiac_data\\33case_training_pair.txt'
test_pair_path = 'D:\Code\cardiac_data\\33case_testing_pair.txt'
case_list_path = 'D:\Code\cardiac_data\\33case_list.txt'
training_pair_array = np.loadtxt(training_pair_path).astype(np.int32)
test_pair_array = np.loadtxt(test_pair_path).astype(np.int32)
all_pair = np.r_[training_pair_array, test_pair_array]


# %%
def max_min_point_dist(cp_pos):
    cp_num = cp_pos.shape[0]

    bigeye = np.eye(cp_num, cp_num) * 10e4

    dist = np.linalg.norm(
        np.expand_dims(cp_pos, 0) - np.expand_dims(cp_pos, 1), axis=2) + bigeye
    min_dist = np.min(dist, axis=1)
    return np.max(min_dist, axis=0)


def flow_to_grid(flow, step, plt):
    img_h = flow.shape[1]
    img_w = flow.shape[2]
    grid = np.stack(np.meshgrid(np.arange(img_h), np.arange(img_w)),
                    2).transpose(2, 0, 1)
    flow = grid - flow
    for i in range(0, img_h, step):
        line_point = flow[:, i, :]
        plt.plot(line_point[0, :], line_point[1, :], c='red', linewidth=1)
    for i in range(0, img_w, step):
        line_point = flow[:, :, i]
        plt.plot(line_point[0, :], line_point[1, :], c='red', linewidth=1)


def get_NetGI_cp_pos():
    cp_loc_vectors = [
        np.linspace(s + (e - s) / 16, e - (e - s) / 16, 8)
        for s, e in ((0, 128), (0, 128))
    ]
    cp_loc = np.meshgrid(*cp_loc_vectors)
    cp_loc = np.stack(cp_loc, 2)[:, :, [1, 0]]
    cp_loc = np.reshape(cp_loc, [-1, 2]).astype(float)
    lcp_loc_vectors = [
        np.linspace(s + (e - s) / 20, e - (e - s) / 20, 10)
        for s, e in ((2 * 16, 6 * 16), (2 * 16, 6 * 16))
    ]
    lcp_loc = np.meshgrid(*lcp_loc_vectors)
    lcp_loc = np.stack(lcp_loc, 2)[:, :, [1, 0]]
    lcp_loc = np.reshape(lcp_loc, [-1, 2]).astype(float)
    cp_loc = np.concatenate((cp_loc, lcp_loc), 0)
    return cp_loc


def data_generator(batch_size):
    P_list, Q_list = [], []
    for c_no, slc, d, s in all_pair:
        if not check_accept(c_no, slc, d, s):
            continue
        data = get_orignal_data(c_no, slc, d, s)
        P, Q = data[2], data[3]
        P_list.append(P)
        Q_list.append(Q)
        if len(P_list) == batch_size:
            P_list = np.stack(P_list, 0)
            Q_list = np.stack(Q_list, 0)
            P_tensor = torch.from_numpy(P_list).float().cuda()
            Q_tensor = torch.from_numpy(Q_list).float().cuda()
            yield P_tensor, Q_tensor
            P_list, Q_list = [], []

    if len(P_list) > 0:
        P_list = np.stack(P_list, 0)
        Q_list = np.stack(Q_list, 0)
        P_tensor = torch.from_numpy(P_list).float().cuda()
        Q_tensor = torch.from_numpy(Q_list).float().cuda()
        yield P_tensor, Q_tensor


# %%
alpha_list = []
lambda1 = 0
control_points = get_NetGI_cp_pos()
control_points = torch.tensor(control_points).cuda().float()

bs = 196
for P, Q in data_generator(bs):
    batch_size = P.size()[0]
    c = torch.tensor([[[16, 24, 32]]]).cuda()
    old_alpha = CoeffMultiC(P, Q, c, 0, 0, 200000)
    moved_P = P + move(P, P, old_alpha, c)

    # flow = torch.zeros((2, 128, 128))
    # for x in range(0, 128):
    #     points = []
    #     for y in range(0, 128):
    #         points.append([x, y])
    #     points = torch.tensor(np.array(points)).unsqueeze(0).cuda()
    #     flow[:, :, x] = -move(points, P[:1], old_alpha[:1], c)[0].permute(
    #         1, 0).cpu()
    # flow = flow.numpy()
    # flow_to_grid(flow, 3, plt)
    # plt.xlim(0, 128)
    # plt.ylim(128, 0)
    # plt.show()

    batch = P.size()[0]
    cp_batch = control_points.unsqueeze(0).repeat(batch_size, 1, 1)
    moved_cp = control_points + move(cp_batch, P, old_alpha, c)
    c = max_min_point_dist(get_NetGI_cp_pos()) * torch.tensor(
        [[[1.2, 1.5, 1.8]]]).cuda()
    c = torch.tensor([[[16, 18, 20]]]).cuda()

    new_alpha = CoeffMultiC(cp_batch, moved_cp, c, lambda1, 0, 500000)

    # flow = torch.zeros((2, 128, 128))
    # for x in range(0, 128):
    #     points = []
    #     for y in range(0, 128):
    #         points.append([x, y])
    #     points = torch.tensor(np.array(points)).unsqueeze(0).cuda()
    #     flow[:, :,
    #          x] = -move(points, cp_batch[:1], new_alpha[:1], c)[0].permute(
    #              1, 0).cpu()
    # flow = flow.numpy()
    # flow_to_grid(flow, 3, plt)
    # plt.xlim(0, 128)
    # plt.ylim(128, 0)
    # plt.show()

    # break
    alpha_list.append(new_alpha)
# %%
draw_alpha = new_alpha[0].cpu().numpy().reshape(-1, 1)
print(draw_alpha.shape)
plt.bar(np.arange(draw_alpha.shape[0]), draw_alpha[:, 0])
plt.show()
# %%
alpha_tensor = torch.concat(alpha_list, 0)
alpha_array = alpha_tensor.cpu().numpy()
alpha_x = alpha_array[:, :, 0]
alpha_x_group = alpha_x.reshape(-1, 164, 3)
np.savetxt('alpha_x_group.txt', alpha_x_group[1])
alpha_x_max = np.max(np.abs(alpha_x_group), axis=2, keepdims=True)
np.savetxt('alpha_x_max.txt', alpha_x_max[1] == np.abs(alpha_x_group[1]))
alpha_x = alpha_x_group * (alpha_x_max == np.abs(alpha_x_group))
np.savetxt("alpha_x0.txt", alpha_x[1].transpose(1, 0))
alpha_x = alpha_x.transpose(0, 2, 1).reshape(-1, 164 * 3)
# alpha_x = np.reshape(alpha_x, (-1, 164, 3)).transpose(0, 2,
#                                                       1).reshape(-1, 164 * 3)
alpha_y = alpha_array[:, :, 1]
alpha_y_group = alpha_y.reshape(-1, 164, 3)
alpha_y_max = np.max(np.abs(alpha_y_group), axis=2, keepdims=True)
alpha_y = alpha_y_group * (alpha_y_max == np.abs(alpha_y_group))
alpha_y = alpha_y.transpose(0, 2, 1).reshape(-1, 164 * 3)
# alpha_y = np.reshape(alpha_y, (-1, 164, 3)).transpose(0, 2,
#                                                       1).reshape(-1, 164 * 3)
cov_x = np.cov(alpha_x, rowvar=False)
np.savetxt("cov_x.txt", cov_x)
cov_y = np.cov(alpha_y, rowvar=False)

cp = get_NetGI_cp_pos()
cp1 = np.expand_dims(cp, 0)
cp2 = np.expand_dims(cp, 1)
dist = np.linalg.norm(cp1 - cp2, axis=2)
distx3 = np.kron(np.eye(3), dist)
c = np.array([[16, 18, 20]])
c = np.repeat(c, 164, axis=1)

distx3_norm = distx3 / c
distx3_norm[distx3_norm > 1] = 1
prior_conv_inverse = np.power(1 - distx3_norm, 4) * (4 * distx3_norm + 1)
prior_conv = np.linalg.inv(prior_conv_inverse)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im = axes[1].imshow(cov_x,
                    cmap='RdBu_r',
                    norm=colors.Normalize(vmin=-0.005, vmax=0.005))
axes[1].set_title('X', fontsize=20)
fig.colorbar(im, ax=axes[1], shrink=0.7, ticks=[-0.005, 0, 0.005])
im = axes[2].imshow(cov_y,
                    cmap='RdBu_r',
                    norm=colors.Normalize(vmin=-0.005, vmax=0.005))
axes[2].set_title('Y', fontsize=20)
fig.colorbar(im, ax=axes[2], shrink=0.7, ticks=[-0.005, 0, 0.005])
im = axes[0].imshow(prior_conv,
                    cmap='RdBu_r',
                    norm=colors.Normalize(vmin=-0.08, vmax=0.08))
axes[0].set_title('Prior', fontsize=20)
fig.colorbar(im, ax=axes[0], shrink=0.7, ticks=[-0.08, 0, 0.08])
for ax in axes:
    ax.set_xticks([0, 164, 164 * 2, 164 * 3])
    ax.set_xticklabels(['1', '164', '328', '492'], fontsize=13)
    ax.set_yticks([0, 164, 164 * 2, 164 * 3])
    ax.set_yticklabels(['1', '164', '328', '492'], fontsize=13)
plt.show()
# %%
