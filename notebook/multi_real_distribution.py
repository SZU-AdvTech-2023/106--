# %%
import sys
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors

# sys.path.append('..')


# %%
def rotation_array(array: np.ndarray, index: int) -> np.ndarray:
    if index == 0: return array
    return np.r_[array[index:], array[:index]]


def correct_array(src_arr: np.ndarray, tgt_arr: np.ndarray) -> np.ndarray:
    rotate_list = []
    for i in range(len(src_arr)):
        rotate_list.append(np.expand_dims(rotation_array(src_arr, i), axis=0))
    rotated_array = np.concatenate(rotate_list, axis=0)

    dist_arr = np.linalg.norm(np.expand_dims(tgt_arr, 0) - rotated_array,
                              axis=2)
    dist_sum_arr = np.sum(dist_arr, axis=1)
    min_idx = np.argmin(dist_sum_arr)
    return rotation_array(src_arr, min_idx)


def handler_seg_point(ed_seg_points: np.ndarray,
                      es_seg_points: np.ndarray) -> np.ndarray:
    ed_lv, ed_bp = ed_seg_points[:32], ed_seg_points[-32:]
    es_lv, es_bp = es_seg_points[:32], es_seg_points[-32:]
    es_lv = correct_array(es_lv, ed_lv)
    es_bp = correct_array(es_bp, ed_bp)
    return np.r_[ed_lv, ed_bp], np.r_[es_lv, es_bp]


def get_center(seg_points: np.ndarray):
    x = int(np.max(seg_points[:, 0]) + np.min(seg_points[:, 0])) // 2
    y = int(np.max(seg_points[:, 1]) + np.min(seg_points[:, 1])) // 2
    return x, y


def csrbf(dist):
    mask = dist < 1
    weight = np.power(1 - dist, 4) * (4 * dist + 1)
    weight = weight * mask.astype(np.float)
    return weight


def get_dvf(pos: np.ndarray, cp_pos: np.ndarray, alpha: np.ndarray, c: int):
    dist = np.linalg.norm(pos - cp_pos, axis=1) / c
    weight = np.expand_dims(csrbf(dist), axis=0)
    return np.matmul(weight, alpha)


def max_min_point_dist(cp_pos):
    cp_num = cp_pos.shape[0]

    bigeye = np.eye(cp_num, cp_num) * 10e4

    dist = np.linalg.norm(
        np.expand_dims(cp_pos, 0) - np.expand_dims(cp_pos, 1), axis=2) + bigeye
    min_dist = np.min(dist, axis=1)
    return np.max(min_dist, axis=0)


def get_csrbf_alpha(cp_pos, X, y, c):
    cp1 = np.expand_dims(cp_pos, 0)
    cp2 = np.expand_dims(X, 1)
    dist = np.linalg.norm(cp1 - cp2, axis=2) / c
    weight_inv = np.linalg.inv(csrbf(dist))
    # print(np.linalg.det(csrbf(dist)))
    # print(np.linalg.det(weight_inv))
    # plt.imshow(dist)
    # plt.show()
    # plt.imshow(csrbf(dist))
    # plt.show()
    # print(weight_inv)
    # print(np.max(weight_inv), np.min(weight_inv))
    # plt.imshow(weight_inv)
    # plt.show()
    alpha = np.matmul(weight_inv, y)
    return alpha


def get_multicsrbf_alpha(cp_pos, x, y, c_list):
    cp1 = np.expand_dims(cp_pos, 0)
    cp2 = np.expand_dims(x, 1)
    weight_list = []
    for c in c_list:
        dist = np.linalg.norm(cp1 - cp2, axis=2) / c
        weight_list.append(csrbf(dist))
    weight = np.concatenate(weight_list, axis=1)
    weight_inv = np.linalg.inv(weight)
    alpha = np.matmul(weight_inv, y)
    return alpha


def get_NetGI_cp_pos():
    cp_loc_vectors = [
        np.linspace(s + (e - s) / 16, e - (e - s) / 16, 8)
        for s, e in ((0, 127), (0, 127))
    ]
    cp_loc = np.meshgrid(*cp_loc_vectors)
    cp_loc = np.stack(cp_loc, 2)[:, :, [1, 0]]
    cp_loc = np.reshape(cp_loc, [-1, 2]).astype(np.float)
    lcp_loc_vectors = [
        np.linspace(s + (e - s) / 20, e - (e - s) / 20, 10)
        for s, e in ((2 * 16 - 1, 6 * 16 - 1), (2 * 16 - 1, 6 * 16 - 1))
    ]
    lcp_loc = np.meshgrid(*lcp_loc_vectors)
    lcp_loc = np.stack(lcp_loc, 2)[:, :, [1, 0]]
    lcp_loc = np.reshape(lcp_loc, [-1, 2]).astype(np.float)
    cp_loc = np.concatenate((cp_loc, lcp_loc), 0)
    return cp_loc


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


def check_accept(case, slice_no, ed, es):
    seg_path = os.path.join('D:\Code\cardiac_data\orginal_data\\33Cases\\seg',
                            'manual_seg_32points_pat%d.mat' % (case))
    seg_zt = sio.loadmat(seg_path)['manual_seg_32points']
    ed_seg_point = seg_zt[slice_no - 1, ed] - 1
    es_seg_point = seg_zt[slice_no - 1, es] - 1
    return ed_seg_point.shape[0] > 32 and es_seg_point.shape[0] > 32


def get_orignal_data(case, slice_no, ed, es, cp_pos=None):
    image_path = os.path.join(
        'D:\Code\cardiac_data\orginal_data\\33Cases\\vol',
        'sol_yxzt_pat%d.mat' % (case))
    seg_path = os.path.join('D:\Code\cardiac_data\orginal_data\\33Cases\\seg',
                            'manual_seg_32points_pat%d.mat' % (case))
    image_yxzt = sio.loadmat(image_path)['sol_yxzt']
    ed_image = image_yxzt[:, :, slice_no - 1, ed]
    es_image = image_yxzt[:, :, slice_no - 1, es]
    seg_zt = sio.loadmat(seg_path)['manual_seg_32points']
    ed_seg_point = seg_zt[slice_no - 1, ed] - 1
    es_seg_point = seg_zt[slice_no - 1, es] - 1
    ed_seg_point, es_seg_point = handler_seg_point(ed_seg_point, es_seg_point)
    # crop
    center_x, center_y = get_center(ed_seg_point)
    ed_seg_point = ed_seg_point - np.array([center_x, center_y]) + 64
    es_seg_point = es_seg_point - np.array([center_x, center_y]) + 64
    ed_image = ed_image[center_y - 64:center_y + 64,
                        center_x - 64:center_x + 64]
    es_image = es_image[center_y - 64:center_y + 64,
                        center_x - 64:center_x + 64]

    displace_vector = ed_seg_point - es_seg_point

    fixed_cp = np.array([[0, 127], [0, 0], [127, 0], [127, 127], [0, 64],
                         [64, 0], [127, 64], [64, 127]])
    zero_dvf = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                         [0, 0], [0, 0]])

    ed_seg_point = np.r_[ed_seg_point, fixed_cp]
    displace_vector = np.r_[displace_vector, zero_dvf]
    return ed_image, es_image, ed_seg_point, displace_vector


def get_real_alpha(case, slice_no, ed, es, cp_pos=None):
    image_path = os.path.join(
        'D:\Code\cardiac_data\orginal_data\\33Cases\\vol',
        'sol_yxzt_pat%d.mat' % (case))
    seg_path = os.path.join('D:\Code\cardiac_data\orginal_data\\33Cases\\seg',
                            'manual_seg_32points_pat%d.mat' % (case))
    image_yxzt = sio.loadmat(image_path)['sol_yxzt']
    ed_image = image_yxzt[:, :, slice_no - 1, ed]
    es_image = image_yxzt[:, :, slice_no - 1, es]
    seg_zt = sio.loadmat(seg_path)['manual_seg_32points']
    ed_seg_point = seg_zt[slice_no - 1, ed] - 1
    es_seg_point = seg_zt[slice_no - 1, es] - 1
    ed_seg_point, es_seg_point = handler_seg_point(ed_seg_point, es_seg_point)
    # crop
    center_x, center_y = get_center(ed_seg_point)
    ed_seg_point = ed_seg_point - np.array([center_x, center_y]) + 64
    es_seg_point = es_seg_point - np.array([center_x, center_y]) + 64
    ed_image = ed_image[center_y - 64:center_y + 64,
                        center_x - 64:center_x + 64]
    es_image = es_image[center_y - 64:center_y + 64,
                        center_x - 64:center_x + 64]

    displace_vector = ed_seg_point - es_seg_point
    # imshow for check
    # plt.figure(figsize=(20, 20))
    # plt.imshow(ed_image, 'gray')
    # plt.scatter(ed_seg_point[:, 0], ed_seg_point[:, 1])
    # plt.scatter(es_seg_point[:, 0], es_seg_point[:, 1])
    # plt.quiver(es_seg_point[:, 0],
    #            es_seg_point[:, 1],
    #            displace_vector[:, 0],
    #            displace_vector[:, 1],
    #            angles='xy',
    #            scale=1,
    #            scale_units='xy',
    #            color='blue',
    #            width=0.001)
    # plt.show()

    fixed_cp = np.array([[0, 127], [0, 0], [127, 0], [127, 127], [0, 64],
                         [64, 0], [127, 64], [64, 127]])
    zero_dvf = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                         [0, 0], [0, 0]])

    ed_seg_point = np.r_[ed_seg_point, fixed_cp]
    displace_vector = np.r_[displace_vector, zero_dvf]

    c = max_min_point_dist(ed_seg_point) * 2
    old_alpha = get_csrbf_alpha(ed_seg_point, ed_seg_point, displace_vector, c)
    # return old_alpha

    if cp_pos is not None:
        new_cp_pos = cp_pos
    else:
        new_cp_pos = get_NetGI_cp_pos()
    # X = np.random.uniform(0, 127, size=(new_cp_pos.shape[0] * 3, 2))
    # X = np.random.uniform(0, 127, size=(new_cp_pos.shape[0] - 8, 2))
    X1 = new_cp_pos + 1
    X2 = new_cp_pos - 1
    X = np.concatenate([new_cp_pos, X1, X2], axis=0)
    new_displace_vector = np.zeros_like(X)
    # print(X)
    for i, cp in enumerate(X):
        new_displace_vector[i] = get_dvf(cp, ed_seg_point, old_alpha, c)
    # print(new_displace_vector)
    # X = np.r_[X, fixed_cp]
    # new_displace_vector = np.r_[new_displace_vector, zero_dvf]
    new_c = max_min_point_dist(new_cp_pos) * np.array([1.5, 2, 2.5])
    # new_c = max_min_point_dist(new_cp_pos) * 2
    # print(new_c)
    new_alpha = get_multicsrbf_alpha(new_cp_pos, X, new_displace_vector, new_c)
    # new_alpha = get_csrbf_alpha(new_cp_pos, X, new_displace_vector, new_c)
    val_displace_vector = np.zeros_like(X)
    # for i, cp in enumerate(X):
    #     val_displace_vector[i] = get_dvf(cp, new_cp_pos, new_alpha, new_c)
    # print(val_displace_vector)
    flow = np.zeros((2, 128, 128))
    # for i in range(0, 128):
    #     for j in range(0, 128):
    #         flow[:, i, j] = get_dvf(np.array([[j, i]]), new_cp_pos, new_alpha,
    #                                 new_c)
    #         # flow[:, i, j] = get_dvf(np.array([[j, i]]), ed_seg_point,
    #         #                         old_alpha, c)
    # # plt.scatter(new_cp_pos[:, 0], new_cp_pos[:, 1])
    # plt.scatter(X[:, 0], X[:, 1])
    # flow_to_grid(flow, 3, plt)
    # plt.xlim(0, 128)
    # plt.ylim(128, 0)
    # plt.show()
    return new_alpha


def saveImage(copy_plt, save_path, format='png'):
    # copy_plt.axis('off')
    # copy_plt.gca().set_axis_off()
    copy_plt.subplots_adjust(top=1,
                             bottom=0,
                             right=1,
                             left=0,
                             hspace=0,
                             wspace=0)
    # copy_plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # copy_plt.gca().yaxis.set_major_locator(plt.NullLocator())
    copy_plt.margins(0, 0)
    copy_plt.savefig(save_path,
                     bbox_inches='tight',
                     pad_inches=0,
                     format=format)


# %%
training_pair_path = 'D:\Code\cardiac_data\\33case_training_pair.txt'
test_pair_path = 'D:\Code\cardiac_data\\33case_testing_pair.txt'
case_list_path = 'D:\Code\cardiac_data\\33case_list.txt'
training_pair_array = np.loadtxt(training_pair_path).astype(np.int32)
test_pair_array = np.loadtxt(test_pair_path).astype(np.int32)
all_pair = np.r_[training_pair_array, test_pair_array]
# %%
alpha_list = []
for c, slc, d, s in all_pair:
    if (check_accept(c, slc, d, s)):
        alpha_list.append(get_real_alpha(c, slc, d, s))
        # print(alpha_list)
        # break
alpha = np.stack(alpha_list, 0)
alpha_x = alpha[:, :, 0]
alpha_y = alpha[:, :, 1]
cov_X = np.cov(np.concatenate([alpha_x, alpha_y], 1), rowvar=False)

# %%

plt.figure(figsize=[10, 10])
ax = plt.gca()
im = ax.imshow(cov_X,
               norm=colors.Normalize(
                   vmin=-100,
                   vmax=100,
               ),
               cmap='RdBu_r')
plt.yticks([0, 327], labels=[r'$1$', r'$328$'], fontsize=24)
plt.xticks([0, 327], labels=[r'$1$', r'$328$'], fontsize=24)
plt.show()
# %%
alpha.shape

# %%
print(1, alpha[0])
print(2, alpha[1][1])
# %%

# %%
