config_code = '''
import torch

class Config:
    """全局配置类"""
    
    # 设备配置
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 数据配置
    DATA_PATH = './data/raw/'
    FMRI_FILE = 'ADNI_fmri.mat'
    DTI_FILE = 'ADNI_dti.mat'
    
    # 数据预处理
    N_SUBJECTS = 217  # 总样本数
    N_REGIONS = 90    # 脑区数量
    
    # 模型配置
    N_CLASS = 2
    N_FEAT = 90       # 输入特征维度
    N_HID = 90        # 隐藏层维度
    DROPOUT = 0.5
    ALPHA = 0.2       # LeakyReLU负斜率
    BETA = 1e7        # 对比损失权重
    
    # 训练配置
    BATCH_SIZE_TRAIN = 20
    BATCH_SIZE_TEST = 13
    LEARNING_RATE = 2e-5
    EPOCHS = 300
    PATIENCE = 60     # 早停耐心值
    
    # 交叉验证配置
    K_FOLDS = 10
    RANDOM_SEED = 7
    
    # 注意力机制配置
    EMBED_DIM = 8100  # 嵌入维度
    NUM_HEADS = 8100  # 注意力头数
    HIDDEN_SIZE = 90  # 注意力隐藏层大小
    
    # 对比学习配置
    TAU = 0.4         # 温度参数
    PROJECTION_DIM1 = 2048
    PROJECTION_DIM2 = 256
    
    # PageRank配置
    PAGERANK_ALPHA = 0.85
    
    # 分类器配置
    CLASSIFIER_DIMS = [8100, 1024, 128, 2]
    
    # 日志配置
    LOG_FILE = './results/logs/training.log'
    MODEL_SAVE_PATH = './results/models/'
    
    @classmethod
    def get_device_info(cls):
        """获取设备信息"""
        if torch.cuda.is_available():
            return f"CUDA Device: {torch.cuda.get_device_name(0)}"
        else:
            return "CPU Device"
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print("Configuration Settings:")
        print("=" * 50)
        for attr in dir(cls):
            if not attr.startswith('_') and not callable(getattr(cls, attr)):
                print(f"{attr}: {getattr(cls, attr)}")
        print("=" * 50)
'''

with open('/home/user/config/config.py', 'w') as f:
    f.write(config_code)
    
print("config/config.py 创建完成")