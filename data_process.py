import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
import torch

# 数据加载和预处理
def load_data():
    m = loadmat('')  # fmri
    keysm = list(m.keys())
    feature = m[keysm[3]][0:217]  # 特征数据
    net_all = []
    for i in range(feature.shape[0]):
        net = np.corrcoef(feature[i])
        net_all.append(net)
    fdata = np.array(net_all)

    labels = m[keysm[4]][0][0:217]

    n = loadmat('')  # DTI
    keysn = list(n.keys())
    ddata = n[keysn[3]]  # 结构数据
    ddata = ddata.transpose(2, 0, 1)
    ddata = ddata[0:217]

    # 打乱数据
    index = [i for i in range(fdata.shape[0])]
    np.random.shuffle(index)
    fdata = fdata[index]
    labels = labels[index]
    ddata = ddata[index]
    
    return fdata, labels, ddata

# 数据集定义
class Dianxian(Dataset):
    def __init__(self):
        super(Dianxian, self).__init__()
        fdata, labels, ddata = load_data()
        self.nodes = fdata
        self.edges = ddata
        self.labels = labels

    def __getitem__(self, item):
        node = self.nodes[item]
        edge = self.edges[item]
        label = self.labels[item]
        return node, edge, label

    def __len__(self):
        return self.nodes.shape[0]