# %%
from linecache import checkcache
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sqlalchemy import false

from coeff_multi import CoeffMultiC, move
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
    cp_loc = np.reshape(cp_loc, [-1, 2]).astype(np.float)
    lcp_loc_vectors = [
        np.linspace(s + (e - s) / 20, e - (e - s) / 20, 10)
        for s, e in ((2 * 16, 6 * 16), (2 * 16, 6 * 16))
    ]
    lcp_loc = np.meshgrid(*lcp_loc_vectors)
    lcp_loc = np.stack(lcp_loc, 2)[:, :, [1, 0]]
    lcp_loc = np.reshape(lcp_loc, [-1, 2]).astype(np.float)
    cp_loc = np.concatenate((cp_loc, lcp_loc), 0)
    return cp_loc


def getFlow(P, Q, c, lbd1, max_iter):
    alpha = CoeffMultiC(P, Q, c, lbd1, 0, max_iter)
    flow = np.zeros((2, 128, 128))
    for x in range(0, 128):
        points = []
        for y in range(0, 128):
            points.append([x, y])
        points = np.array(points)
        flow[:, :, x] = -move(points, P, alpha, c).transpose(1, 0)
    return alpha, flow


def getBestLambda(l, r, t):
    for c, slc, d, s in all_pair:
        if check_accept(c, slc, d, s):
            data = get_orignal_data(c, slc, d, s)
            break
    P, Q = data[2], data[3]
    c = np.array([[32, 64, 96]])
    alpha, best_flow = getFlow(P, Q, c, l, 100000)
    # flow_to_grid(best_flow, 2, plt)
    # plt.show()
    new_P = get_NetGI_cp_pos()
    new_Q = new_P + move(new_P, P, alpha, c)
    # print(move(new_P, P, alpha, c))
    new_c = max_min_point_dist(control_points) * np.array([1.5, 2, 2.5])
    new_c = np.expand_dims(new_c, 0)

    while r - l > 1e-11:
        mid = (r + l) / 2
        # mid = 0
        _, flow = getFlow(new_P, new_Q, new_c, mid, 100000)
        # flow_to_grid(flow, 2, plt)
        # plt.show()
        diff = np.sum(best_flow - flow)
        if diff <= t:
            l = mid + 1e-11
        else:
            r = mid - 1e-11
        print(diff, l, r)
        # break
    return l


def getBestIter(l, r, t):
    for c, slc, d, s in all_pair:
        if check_accept(c, slc, d, s):
            data = get_orignal_data(c, slc, d, s)
            break
    P, Q = data[2], data[3]
    c = np.array([[32, 64, 96]])
    alpha, _ = getFlow(P, Q, c, 0, 100000)
    # flow_to_grid(best_flow, 2, plt)
    # plt.show()
    new_P = get_NetGI_cp_pos()
    new_Q = new_P + move(new_P, P, alpha, c)
    # print(move(new_P, P, alpha, c))
    new_c = max_min_point_dist(control_points) * np.array([1.5, 2, 2.5])
    new_c = np.expand_dims(new_c, 0)
    _, best_flow = getFlow(new_P, new_Q, new_c, 5e-7, r)

    while l < r:
        mid = (r + l) // 2
        # mid = 0
        _, flow = getFlow(new_P, new_Q, new_c, 5e-7, mid)
        # flow_to_grid(flow, 2, plt)
        # plt.show()
        diff = np.sum(best_flow - flow)
        if diff <= t:
            r = mid - 1
        else:
            l = mid + 1
        print(diff, l, r)
        # break
    return r


# print(getBestLambda(0, 1, 1e3))
# print(getBestIter(1, 100000, 1e2))

# %%
alpha_list = []
lambda1 = 0
control_points = get_NetGI_cp_pos()
# control_points = np.random.uniform(0, 127, size=(32, 2))
# np.random.shuffle(all_pair)
for c, slc, d, s in all_pair:
    if check_accept(c, slc, d, s):
        data = get_orignal_data(c, slc, d, s)
        P, Q = data[2], data[3]
        c = max_min_point_dist(P) * np.array([1.5, 2, 2.5])
        c = np.array([16, 32, 64])
        c = np.array([24])
        c = np.expand_dims(c, 0)
        old_alpha = CoeffMultiC(P, Q, c, 0, 0, 200000)
        moved_P = P + move(P, P, old_alpha, c)
        print(np.abs(moved_P - Q).sum())
        flow = np.zeros((2, 128, 128))
        for x in range(0, 128):
            points = []
            for y in range(0, 128):
                points.append([x, y])
            points = np.array(points)
            flow[:, :, x] = -move(points, P, old_alpha, c).transpose(1, 0)
        flow_to_grid(flow, 3, plt)
        plt.xlim(0, 128)
        plt.ylim(128, 0)
        plt.show()
        moved_cp = control_points + move(control_points, P, old_alpha, c)
        c = max_min_point_dist(control_points) * np.array([1.5, 2, 2.5])
        c = max_min_point_dist(control_points) * np.array([1])
        c = np.expand_dims(c, 0)
        new_alpha = CoeffMultiC(control_points, moved_cp, c, lambda1, 0,
                                500000)
        draw_alpha = np.reshape(np.abs(new_alpha), [-1])
        plt.bar(np.arange(draw_alpha.shape[0]), draw_alpha)
        plt.show()
        flow = np.zeros((2, 128, 128))
        for x in range(0, 128):
            points = []
            for y in range(0, 128):
                points.append([x, y])
            points = np.array(points)
            flow[:, :,
                 x] = -move(points, control_points, new_alpha, c).transpose(
                     1, 0)
        flow_to_grid(flow, 3, plt)
        plt.xlim(0, 128)
        plt.ylim(128, 0)
        plt.show()
        # print((np.abs(new_alpha) < 1e-4).sum())
        break
        alpha_list.append(new_alpha)

# %%
# single C
alpha_single = np.stack(alpha_list, 0)
cov_X = np.cov(alpha_single[:, :, 0], rowvar=False)
cov_Y = np.cov(alpha_single[:, :, 1], rowvar=False)
max_idx = np.argmax(cov_X[:40])
print(max_idx // 164, max_idx % 164)
p1, p2 = max_idx // 164, max_idx % 164
print(control_points[p1, :], control_points[p2, :])

fig, axes = plt.subplots(1, 2, figsize=[10, 10])
im = axes[0].imshow(cov_X, cmap='RdBu_r')
axes[0].set_title('X')
plt.colorbar(im, ax=axes[0], shrink=0.3)
im = axes[1].imshow(cov_Y, cmap='RdBu_r')
axes[1].set_title('Y')
plt.colorbar(im, ax=axes[1], shrink=0.3)
plt.show()
# %%
alpha = np.stack(alpha_list, 0)
alpha = np.reshape(alpha, (alpha.shape[0], alpha.shape[1] // 3, 3, 2))
alpha = np.transpose(alpha, (0, 2, 3, 1))
alpha = np.reshape(alpha, (alpha.shape[0], -1))
cov_X = np.cov(alpha, rowvar=False)
print((np.abs(cov_X) < 1e-5).sum(), cov_X.shape[0] * cov_X.shape[1])
print((np.abs(alpha) < 1e-5).sum(), alpha.shape[0] * alpha.shape[1])
# %%
alpha_draow = np.reshape(alpha, (-1, 3, 2, 164))
alpha_x = np.reshape(alpha_draow[:, :, 0, :], [alpha.shape[0], -1])
alpha_y = np.reshape(alpha_draow[:, :, 1, :], [alpha.shape[0], -1])
cov_x = np.cov(alpha_x, rowvar=False)
cov_y = np.cov(alpha_y, rowvar=False)

fig, axes = plt.subplots(1, 2, figsize=[10, 10])
im = axes[0].imshow(cov_x, cmap='RdBu_r')
axes[0].set_title('X')
plt.colorbar(im, ax=axes[0], shrink=0.3)
im = axes[1].imshow(cov_y, cmap='RdBu_r')
axes[1].set_title('Y')
plt.colorbar(im, ax=axes[1], shrink=0.3)
plt.show()
# %%

plt.figure(figsize=[10, 10])
ax = plt.gca()
im = ax.imshow(
    cov_X,
    #    norm=colors.Normalize(
    #        vmin=-1,
    #        vmax=1,
    #    ),
    cmap='RdBu_r')
plt.yticks([0, 328 * 3 - 1],
           labels=[r'$1$', r'$%d$' % (328 * 3 - 1)],
           fontsize=24)
plt.xticks([0, 328 * 3 - 1],
           labels=[r'$1$', r'$%d$' % (328 * 3 - 1)],
           fontsize=24)
plt.show()

# %%
cp = get_NetGI_cp_pos()
cp1 = np.expand_dims(cp, 0)
cp2 = np.expand_dims(cp, 1)
dist = np.linalg.norm(cp1 - cp2, axis=2)
distxyx3 = np.kron(np.eye(6), dist)

c = max_min_point_dist(control_points) * np.array([1.5, 2, 2.5])
c = np.repeat(c, cp.shape[0] * 2)
c = np.expand_dims(c, 0)

distxyx3_norm = distxyx3 / c
distxyx3_norm[distxyx3_norm > 1] = 1
prior_conv_inverse = np.power(1 - distxyx3_norm, 4) * (4 * distxyx3_norm + 1)
# print(distxyx3_norm)
prior_conv = np.linalg.inv(prior_conv_inverse)
print(prior_conv.shape)
# %%
plt.figure(figsize=[10, 10])
ax = plt.gca()
im = ax.imshow(prior_conv,
               norm=colors.Normalize(
                   vmin=-1,
                   vmax=1,
               ),
               cmap='RdBu_r')
plt.yticks([0, 328 * 3 - 1],
           labels=[r'$1$', r'$%d$' % (328 * 3 - 1)],
           fontsize=24)
plt.xticks([0, 328 * 3 - 1],
           labels=[r'$1$', r'$%d$' % (328 * 3 - 1)],
           fontsize=24)
plt.show()
# %%
