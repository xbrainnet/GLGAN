import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from config.config import Config

class BrainDataset(Dataset):
    """脑网络数据集"""
    
    def __init__(self, fmri_data, dti_data, labels, indices=None):
        """
        初始化数据集
        
        Args:
            fmri_data: fMRI功能连接数据
            dti_data: DTI结构连接数据  
            labels: 标签
            indices: 样本索引
        """
        if indices is not None:
            self.fmri_data = fmri_data[indices]
            self.dti_data = dti_data[indices] 
            self.labels = labels[indices]
        else:
            self.fmri_data = fmri_data
            self.dti_data = dti_data
            self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'fmri': torch.FloatTensor(self.fmri_data[idx]),
            'dti': torch.FloatTensor(self.dti_data[idx]),
            'label': torch.LongTensor([self.labels[idx]])
        }

def load_data():
    """加载原始数据"""
    # 加载fMRI数据
    fmri_data = sio.loadmat(Config.DATA_PATH + Config.FMRI_FILE)
    X_data = fmri_data['X_data_gnd']
    
    # 加载DTI数据
    dti_data = sio.loadmat(Config.DATA_PATH + Config.DTI_FILE)
    G_all = dti_data['G_all']
    
    # 提取标签
    labels = X_data[:, -1].astype(int) - 1  # 转换为0,1标签
    fmri_features = X_data[:, :-1]
    
    return fmri_features, G_all, labels

def create_data_loaders(fold_idx=0):
    """
    创建数据加载器
    
    Args:
        fold_idx: 交叉验证折数索引
        
    Returns:
        train_loader, test_loader: 训练和测试数据加载器
    """
    fmri_data, dti_data, labels = load_data()
    
    # 交叉验证分割
    skf = StratifiedKFold(n_splits=Config.K_FOLDS, shuffle=True, 
                         random_state=Config.RANDOM_SEED)
    
    folds = list(skf.split(fmri_data, labels))
    train_idx, test_idx = folds[fold_idx]
    
    # 创建数据集
    train_dataset = BrainDataset(fmri_data, dti_data, labels, train_idx)
    test_dataset = BrainDataset(fmri_data, dti_data, labels, test_idx)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE_TRAIN,
        shuffle=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE_TEST, 
        shuffle=False
    )
    
    return train_loader, test_loader
