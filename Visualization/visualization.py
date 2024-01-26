# %%
import sys
import os
import copy

from numpy.lib.function_base import angle

sys.path.append('..\\')
from solver import Solver
from config import config
import torch
from matplotlib import colors, scale
import numpy as np
import matplotlib.pyplot as plt
from Utils import flow2Grid, flow2img, saveImage
# %%
krebs_save_path = 'D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\KrebsDiff\ECCV-LCC60000[9, 9]-4-20220219195401'
dalca_save_path = 'D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\DalcaDiff\ECCV-MSE-200000-11599.9999999999998-20220220114051'
vm_save_path = 'D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\VoxelMorph\ECCV-LCC1[9, 9]-20220220113300'
gi_save_path = 'D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\RBFGI\ECCV-WLCC---0.02-[9, 9]--130000-20220221001458'
mc_save_path = 'D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\RBFGIMutilAdaptive\ECCV-WLCC---0.02-[9, 9]--1300001502-20220220114330'
config['dataset'] = {
    'training_list_path': 'G:\\cardiac_data\\M&M_training_pair.txt',
    'testing_list_path': 'G:\\cardiac_data\\M&M_testing_pair.txt',
    'validation_list_path': 'G:\\cardiac_data\\M&M_validation_pair.txt',
    'pair_dir': 'G:\\cardiac_data\\2Dwithoutcenter1/',
    'resolution_path': 'G:\\cardiac_data\\resolution.txt'
}

network_list = [
    'KrebsDiff', 'DalcaDiff', 'VoxelMorph', 'RBFGI', 'RBFGIMutilAdaptive'
]
model_save_path_list = [
    krebs_save_path, dalca_save_path, vm_save_path, gi_save_path, mc_save_path
]


# %%
def getSolver(network, model_save_path):
    config['mode'] = 'Test'
    config['network'] = network
    config['Test']['model_save_path'] = model_save_path
    sol = Solver(copy.deepcopy(config))
    # sol.getTestDataloader(sol.config['dataset']['testing_list_path'])
    sol.loadCheckpoint(sol.controller.net, sol.config['Test']['epoch'])
    return sol


# %%
solver_list = [
    getSolver(network, model_save_path)
    for network, model_save_path in zip(network_list, model_save_path_list)
]


# %%
def extractDVF(flow, seg):
    non_zero_pos = np.array(np.nonzero(seg))
    filtered_flow = flow[:, non_zero_pos[0], non_zero_pos[1]]
    return non_zero_pos[[1, 0]].transpose(1, 0), filtered_flow.transpose(1, 0)


def showMyoDVF(c_plt: plt, tgt, flow, seg, step=1):
    non_zero_pos, filtered_flow = extractDVF(flow, seg)
    c_plt.imshow(tgt, cmap=plt.cm.gray)
    dis = np.linalg.norm(non_zero_pos, axis=1)
    # color = plt.cm.spring((dis - 0) / np.max(dis))
    plt.quiver(non_zero_pos[::step, 0] + filtered_flow[::step, 0],
               non_zero_pos[::step, 1] + filtered_flow[::step, 1],
               -filtered_flow[::step, 0],
               -filtered_flow[::step, 1],
               color='yellow',
               angles='xy',
               headwidth=4,
               headlength=6,
               scale_units='xy',
               scale=1,
               width=0.004)


# %%
solver_list[0].getTestDataloader(config['dataset']['testing_list_path'])
dataset = solver_list[0].test_dataloader.dataset
# %%
case_no = 470
alpha = 0.3
ato_no = 2
data = dataset.getByCaseNo(case_no)
slice_num = len(data['slice'])
cmap = colors.ListedColormap([
    (1, 1, 1, 0),  # apparent
    (0, 1, 0, alpha),  # grean for real seg
    (0, 0, 1, alpha),  # blue for predictive seg
    (1, 0, 0, alpha),  # red for overlap
])
plt.figure(figsize=[50, 10 * slice_num])
for i, sol in enumerate(solver_list):
    res = sol.controller.estimate(data)
    tgt = res['tgt'][:, 16:112, 16:112]
    tgt_seg = res['tgt_seg'][:, 16:112, 16:112] == ato_no
    warped_src_seg = res['warped_src_seg'][:, 16:112, 16:112] == ato_no
    warped_src = res['warped_src'][:, 16:112, 16:112]
    overlap = tgt_seg + warped_src_seg * 2
    for j in range(slice_num):
        plt.subplot(slice_num, 5, j * 5 + i + 1)
        plt.imshow(warped_src[j], cmap=plt.cm.gray)
        plt.imshow(overlap[j], cmap=cmap)
plt.show()


# 5 8 10 31 99 101 102 105 107 113 117 136 137 141 156 157* 170 178 456 457
# 458 459 468 501 510 518 519 528
# %%
# 468 5
def drawAll(sol, case_no, slice_no, save_dir=None):
    ato_no = 2
    alpha = 0.3
    cmap = colors.ListedColormap([
        (1, 1, 1, 0),  # apparent
        (0, 1, 0, alpha),  # grean for real seg
        (0, 0, 1, alpha),  # blue for predictive seg
        (1, 0, 0, alpha),  # red for overlap
    ])
    data = dataset.getByCaseNo(case_no)
    res = sol.controller.estimate(data)
    src = res['src'][:, 16:112, 16:112]
    tgt = res['tgt'][:, 16:112, 16:112]
    flow = res['phi'][:, :, 16:112, 16:112]
    tgt_seg = res['tgt_seg'][:, 16:112, 16:112] == ato_no
    warped_src_seg = res['warped_src_seg'][:, 16:112, 16:112] == ato_no
    warped_src = res['warped_src'][:, 16:112, 16:112]
    overlap = tgt_seg + warped_src_seg * 2
    if save_dir:
        save_dir = os.path.join(save_dir, sol.config['network'])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    plt.imshow(src[slice_no], cmap=plt.cm.gray)
    if save_dir:
        saveImage(plt, os.path.join(save_dir, 'src.png'))
    plt.show()
    plt.imshow(tgt[slice_no], cmap=plt.cm.gray)
    if save_dir:
        saveImage(plt, os.path.join(save_dir, 'tgt.png'))
    plt.show()
    plt.imshow(warped_src[slice_no], cmap=plt.cm.gray)
    plt.imshow(overlap[slice_no], cmap=cmap)
    if save_dir:
        saveImage(plt, os.path.join(save_dir, 'warped.png'))
    plt.show()
    plt.imshow(tgt[slice_no], cmap=plt.cm.gray)
    flow2Grid(flow[slice_no], 3, plt, c='yellow')
    if save_dir:
        saveImage(plt, os.path.join(save_dir, 'grid.png'))
    plt.show()
    # flow_img = flow2img(flow[slice_no])
    # plt.imshow(flow_img)
    showMyoDVF(plt, tgt[slice_no], flow[slice_no], tgt_seg[slice_no], step=6)
    if save_dir:
        saveImage(plt, os.path.join(save_dir, 'flow.png'))
    plt.show()


# %%
# 456 2 468 5
save_dir = 'D:\Code\RegistrationPakageForNerualLearning\img'
for sol in solver_list:
    drawAll(sol, 456, 2, save_dir)
# %%
with torch.no_grad():
    net = solver_list[4].controller.net.decoder
    global_cp = net.global_cp_loc * 16
    final_cp = ((net.local_cp_loc - 4) * net.scale + 4) * 16
    orign_cp = net.local_cp_loc * 16
global_cp = global_cp.cpu().numpy()
final_cp = final_cp.cpu().numpy()
orign_cp = orign_cp.cpu().numpy()
# %%
size = 80 
data = dataset.getByCaseNo(529)
src = data['src'][5, 0].cpu().numpy()
tgt = data['tgt'][5, 0].cpu().numpy()
plt.imshow(src, cmap=plt.cm.gray)
plt.scatter(global_cp[:, 0], global_cp[:, 1], marker='x', c='orange', s=size)
plt.scatter(orign_cp[:, 0], orign_cp[:, 1], marker='x', c='yellow', s=size)
saveImage(plt, os.path.join(save_dir, 'src.png'))
plt.show()
plt.imshow(tgt, cmap=plt.cm.gray)
plt.scatter(global_cp[:, 0], global_cp[:, 1], marker='x', c='orange', s=size)
plt.scatter(final_cp[:, 0], final_cp[:, 1], marker='x', c='yellow', s=size)
saveImage(plt, os.path.join(save_dir, 'tgt.png'))
plt.show()
# %%
