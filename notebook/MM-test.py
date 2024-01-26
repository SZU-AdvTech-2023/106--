# %%
from cProfile import label
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

sys.path.append('../')
from config import config
from solver import Solver

matplotlib.use('Agg')

# %%
config['dataset'] = {
    'training_list_path': 'G:\\cardiac_data\\M&M_training_pair.txt',
    'testing_list_path': 'G:\\cardiac_data\\M&M_testing_pair.txt',
    'validation_list_path': 'G:\\cardiac_data\\M&M_validation_pair.txt',
    'pair_dir': 'G:\\cardiac_data\\2Dwithoutcenter1/',
    'resolution_path': 'G:\\cardiac_data\\resolution.txt'
}
config['mode'] = 'Test'
config['network'] = 'RBFGIMutilAdaptivePro'
config['name'] = 'MM-Test'
config['Test'][
    'model_save_path'] = 'D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\RBFGIMutilAdaptivePro\ECCV-onlyMM-WLCC---0.02-[9, 9]--1300001502-20220222154154'
MM_excel_path = 'G:\cardiac_data\orginal_data\OpenDataset\\201014_M&Ms_Dataset_Information_-_opendataset.csv'
MM_test_case_list_path = 'G:\cardiac_data\M&M_test_case_list.txt'
sol = Solver(config)
# %%
mean, details = sol.test()
details['M&M'][438]['LvBp']['ed_to_es']['Dice']
# %%
Code2Vendor = {}
Code2Centre = {}
with open(MM_excel_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        Code2Vendor[row[0]] = row[2]
        Code2Centre[row[0]] = row[3]
with open(MM_test_case_list_path, 'r') as casefile:
    case_list = casefile.readlines()
# %%
vendor_split = {
    'A': {
        'LvMyo': [],
        'Lv': [],
        'Rv': []
    },
    'B': {
        'LvMyo': [],
        'Lv': [],
        'Rv': []
    },
    'C': {
        'LvMyo': [],
        'Lv': [],
        'Rv': []
    },
    'D': {
        'LvMyo': [],
        'Lv': [],
        'Rv': []
    }
}

# %%
for case_no in details['M&M']:
    case_path = case_list[case_no - 438]
    code = case_path.strip().split('\\')[-1]
    vendor = Code2Vendor[code]
    for an in ['LvMyo', 'Rv', 'Lv']:
        d = details['M&M'][case_no][an]['ed_to_es']['Dice']
        vendor_split[vendor][an].append(np.mean(d))

res2str = []
for v in vendor_split:
    for an in ['Lv', 'LvMyo', 'Rv']:
        res2str.append(str(np.round(np.mean(vendor_split[v][an]), 3)))
print('&'.join(res2str))

# %%
box_data = {}
for an in ['LvMyo', 'Rv', 'Lv']:
    box_data[an] = {}
    for m in ['Dice', 'HD']:
        box_data[an][m] = {}
        for i in ['A', 'B', 'C', 'D']:
            box_data[an][m][i] = []
# %%
for case_no in details['M&M']:
    case_path = case_list[case_no - 438]
    code = case_path.strip().split('\\')[-1]
    centre = Code2Vendor[code]
    for an in ['LvMyo', 'Rv', 'Lv']:
        d = details['M&M'][case_no][an]['ed_to_es']['Dice']
        box_data[an]['Dice'][centre].append(np.mean(d))
        d = details['M&M'][case_no][an]['ed_to_es']['HD']
        box_data[an]['HD'][centre].append(np.mean(d))


# %%
def drawBox(plt, an, metric, position, width):
    x, y = [], []
    for k in box_data[an][metric]:
        x.append(k)
        y.append(box_data[an][metric][k])
    box = plt.boxplot(
        y,
        vert=True,
        patch_artist=True,
        #   labels=x,
        showfliers=False,
        positions=position,
        widths=width)
    return box


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3))
bplots = []
for i, a in enumerate(['Lv', 'LvMyo', 'Rv']):
    bp1 = drawBox(axes[0],
                  a,
                  'Dice',
                  position=np.arange(4) * 0.4 - 0.8 + i * 2,
                  width=0.3)
    bp2 = drawBox(axes[1],
                  a,
                  'HD',
                  position=np.arange(4) * 0.4 - 0.8 + i * 2,
                  width=0.3)
    bplots.append(bp1)
    bplots.append(bp2)
axes[0].set_xticks(np.arange(3) * 2)
axes[0].set_xticklabels(['LV', 'MYO', 'RV'], fontsize=12)
axes[0].set_ylabel('Dice', fontsize=12)
axes[1].set_xticks(np.arange(3) * 2)
axes[1].set_xticklabels(['LV', 'MYO', 'RV'], fontsize=12)
axes[1].set_ylabel('HD', fontsize=12)
colors = ['#944D3B', '#E06B4F', '#43BEE0', '#94731E', '#E0B138']
vendors = ['A', 'B', 'C', 'D']
for bplot in bplots:
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
for color, vendor in zip(colors, vendors):
    axes[0].plot([], c=color, label=vendor)
    axes[1].plot([], c=color, label=vendor)
axes[0].legend(fontsize=12)
axes[1].legend(fontsize=12)

fig.tight_layout()
plt.subplots_adjust(wspace=0.15,hspace=0)
plt.savefig('MM_vendor_split.eps', format='eps', dpi=1000)  
plt.show()
# %%
