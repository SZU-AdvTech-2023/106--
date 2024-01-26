import os
import scipy.io as sio
import numpy as np


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

    # fixed_cp = np.array([[0, 127], [0, 0], [127, 0], [127, 127], [0, 64],
    #                      [64, 0], [127, 64], [64, 127]])
    # zero_dvf = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
    #                      [0, 0], [0, 0]])

    # ed_seg_point = np.r_[ed_seg_point, fixed_cp]
    # displace_vector = np.r_[displace_vector, zero_dvf]
    return ed_image, es_image, ed_seg_point, es_seg_point, displace_vector


def check_accept(case, slice_no, ed, es):
    seg_path = os.path.join('D:\Code\cardiac_data\orginal_data\\33Cases\\seg',
                            'manual_seg_32points_pat%d.mat' % (case))
    seg_zt = sio.loadmat(seg_path)['manual_seg_32points']
    ed_seg_point = seg_zt[slice_no - 1, ed] - 1
    es_seg_point = seg_zt[slice_no - 1, es] - 1
    return ed_seg_point.shape[0] > 32 and es_seg_point.shape[0] > 32