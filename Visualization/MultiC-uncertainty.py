# %%
import sys
import os

sys.path.append("..\\")
from config import config
from solver import Solver
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from Utils import saveImage, flow2Grid
import numpy as np

# %%
config['dataset'] = {
    'training_list_path': 'G:\\cardiac_data\\M&M_training_pair.txt',
    'testing_list_path': 'G:\\cardiac_data\\M&M_testing_pair.txt',
    'validation_list_path': 'G:\\cardiac_data\\M&M_validation_pair.txt',
    'pair_dir': 'G:\\cardiac_data\\2Dwithoutcenter1/',
    'resolution_path': 'G:\\cardiac_data\\resolution.txt'
}
config["network"] = "RBFGIMutilAdaptivePro"
config["Test"][
    "model_save_path"] = "D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\RBFGIMutilAdaptivePro\ECCV-onlyMM-WLCC---0.02-[9, 9]--1300001502-20220222154154"
config["mode"] = "Test"
config["RBFGIMutilAdaptive"]["params"]["int_steps"] = None
sol = Solver(config)
sol.loadCheckpoint(sol.controller.net, sol.config["Test"]["epoch"])
# %%
sol.getTestDataloader(sol.config["dataset"]["testing_list_path"])
dataset = sol.test_dataloader.dataset
case_list = np.array(dataset.case_list)
case_list = case_list[(case_list < 33) + (case_list > 78)]
# %%
image_num = 4
row = 6
K = 500
np.random.seed(3)
case_no_list = np.random.choice(case_list, size=[image_num])
img_dir = "D:\Code\RegistrationPakageForNerualLearning\img\\uncertainty"
plt.figure(figsize=(image_num * 10, 10 * row))
for i, case_no in enumerate(case_no_list):
    # plt.figure(figsize=(15, 15))
    data = dataset.getByCaseNo(case_no)
    res = sol.controller.estimate(data)
    src = data["src"].cuda().float()
    tgt = data["tgt"].cuda().float()
    slice_num = len(data["slice"])
    slice_no = np.random.choice(range(slice_num), size=1)[0]
    with torch.no_grad():
        res = sol.controller.net.uncertainty(src, tgt, K)
        cpoint_pos = sol.controller.net.decoder.cp_gen(sol.controller.net.scale)
        uncertain = res[0][slice_no]
        uncertain = res[0].cpu().numpy()[slice_no]
        phi = res[1].cpu().numpy()[slice_no]
        cpoint_pos = cpoint_pos.cpu().numpy() * 16  # [N 2]
        log_var = []
        # for lv in res[2]:
        #     # log_var.append(torch.sqrt(torch.exp(lv)).cpu().numpy()[slice_no])
        #     log_var.append(lv.cpu().numpy()[slice_no])
        #     log_var[-1][:, 0] = (log_var[-1][:, 0] - np.mean(
        #         log_var[-1][:, 0])) / np.std(log_var[-1][:, 0])
        #     log_var[-1][:, 1] = (log_var[-1][:, 1] - np.mean(
        #         log_var[-1][:, 1])) / np.std(log_var[-1][:, 1])


    src = data["src"].cpu().numpy()[slice_no, 0]
    tgt = data["tgt"].cpu().numpy()[slice_no, 0]
    """
    plt.subplot(331)
    plt.imshow(src, cmap=plt.cm.gray)
    plt.title('src')
    plt.axis('off')
    plt.subplot(332)
    plt.imshow(uncertain[2], cmap=plt.cm.gray)
    plt.title('magnitude')
    plt.axis('off')
    plt.subplot(333)
    plt.imshow(uncertain[3], cmap=plt.cm.gray)
    plt.title('angle')
    plt.axis('off')
    for i in range(3):
        plt.subplot(3, 3, i + 4)
        plt.imshow(tgt, cmap=plt.cm.gray)
        plt.axis('off')
        plt.scatter(cpoint_pos[:, 0],
                    cpoint_pos[:, 1],
                    s=100,
                    marker='x',
                    c=log_var[i][:, 0],
                    cmap=plt.cm.spring,
                    linewidths=3)
        plt.title('x%d' % i)
    for i in range(3):
        plt.subplot(3, 3, i + 7)
        plt.imshow(tgt, cmap=plt.cm.gray)
        plt.axis('off')
        plt.scatter(cpoint_pos[:, 0],
                    cpoint_pos[:, 1],
                    s=100,
                    marker='x',
                    c=log_var[i][:, 1],
                    cmap=plt.cm.spring,
                    linewidths=3)
        plt.title('y%d' % i)
    plt.show()
    """
    # """
    plt.imshow(src, cmap=plt.cm.gray)
    saveImage(plt,os.path.join(img_dir,'src-%d.png'%(i)))
    plt.show()
    plt.imshow(tgt, cmap=plt.cm.gray)
    saveImage(plt,os.path.join(img_dir,'tgt-%d.png'%(i)))
    plt.show()
    plt.imshow(uncertain[2], cmap=plt.cm.gray)
    saveImage(plt,os.path.join(img_dir,'unr-%d.png'%(i)))
    plt.show()
    # plt.imshow(tgt, cmap=plt.cm.gray)
    plt.imshow(uncertain[3], cmap=plt.cm.gray)
    saveImage(plt,os.path.join(img_dir,'unt-%d.png'%(i)))
    plt.show()
    plt.imshow(uncertain[0], cmap=plt.cm.gray)
    saveImage(plt,os.path.join(img_dir,'unx-%d.png'%(i)))
    plt.show()
    plt.imshow(uncertain[1], cmap=plt.cm.gray)
    saveImage(plt,os.path.join(img_dir,'uny-%d.png'%(i)))
    plt.show()
    # plt.imshow(uncertain[0],cmap=plt.cm.gray)
    # plt.imshow(uncertain[1],cmap=plt.cm.gray)
    # """
    # unr, unt = uncertain[2], uncertain[3]
    # var, vat = uncertain[6], uncertain[7]

    # diff_r = torch.mean(unr - var)
    # diff_t = torch.mean(unt - vat)
    # print(diff_r, diff_t)
    # print("unr", torch.max(unr), torch.min(unr))
    # print("var", torch.max(var), torch.min(var))
# %%
case_no = case_list[0]
data = dataset.getByCaseNo(case_no)
src = data["src"].cuda().float()
tgt = data["tgt"].cuda().float()
with torch.no_grad():
    res = sol.controller.net.uncertainty(src, tgt, K)[0].cpu().numpy()
    plt.subplot(1, 2, 1)
    plt.imshow(res[0, 2, :, :], cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.imshow(res[0, 4, :, :], cmap=plt.cm.gray)
    plt.show()
# %%
image_num = 4
ato_no = 2
alpha = 0.3
cmap = colors.ListedColormap([
    (1, 1, 1, 0),  # apparent
    (0, 1, 0, alpha),  # grean for real seg
    (0, 0, 1, alpha),  # blue for predictive seg
    (1, 0, 0, alpha),  # red for overlap
])
case_no_list = np.random.choice(case_list, size=image_num)
print(case_no_list)
case_no_list = [502, 79, 480, 457, 516, 99]
img_dir = "D:\Code\RegistrationPakageForNerualLearning\img\meshAndSeg"
for i, case_no in enumerate(case_no_list):
    data = dataset.getByCaseNo(case_no)
    res = sol.controller.estimate(data)
    flow = res["phi"][:, :, 16:112, 16:112]
    warped = res["warped_src"][:, 16:112, 16:112]
    if case_no <= 33:
        ato_no = 1
    tgt_seg = res["tgt_seg"][:, 16:112, 16:112] == ato_no
    warped_src_seg = res["warped_src_seg"][:, 16:112, 16:112] == ato_no
    overlap = tgt_seg + warped_src_seg * 2
    slice_no = np.random.choice(range(tgt.shape[0]), size=1)[0]
    plt.imshow(warped[slice_no], cmap=plt.cm.gray)
    plt.imshow(overlap[slice_no], cmap=cmap)
    saveImage(plt, os.path.join(img_dir, "seg-%d.png" % i))
    plt.show()
    flow2Grid(flow[slice_no], 3, plt, c="yellow")
    plt.imshow(warped[slice_no], cmap=plt.cm.gray)
    saveImage(plt, os.path.join(img_dir, "mesh-%d.png" % i))
    plt.show()
# 502 79 480 457 516 99
# %%
