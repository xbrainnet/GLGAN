seed_code = '''
import random
import numpy as np
import torch

def set_seed(seed=7):
    """
    设置随机种子以确保实验可重复性
    
    Args:
        seed (int): 随机种子值
    """
    torch.manual_seed(seed)           # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)      # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)              # Numpy模块
    random.seed(seed)                 # Python random模块
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")

def get_random_state():
    """获取当前随机状态"""
    return {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'random': random.getstate()
    }

def set_random_state(state):
    """恢复随机状态"""
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['random'])
'''

with open('/home/user/utils/seed.py', 'w') as f:
    f.write(seed_code)
    
print("utils/seed.py 创建完成")
