logger_code = '''
import logging
import os
from datetime import datetime

class Logger:
    """日志记录器"""
    
    def __init__(self, log_file=None, level=logging.INFO):
        """
        初始化日志记录器
        
        Args:
            log_file (str): 日志文件路径
            level: 日志级别
        """
        self.logger = logging.getLogger('GLGAT')
        self.logger.setLevel(level)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        """记录信息"""
        self.logger.info(message)
    
    def warning(self, message):
        """记录警告"""
        self.logger.warning(message)
    
    def error(self, message):
        """记录错误"""
        self.logger.error(message)
    
    def debug(self, message):
        """记录调试信息"""
        self.logger.debug(message)

def create_experiment_logger(experiment_name):
    """
    为实验创建专门的日志记录器
    
    Args:
        experiment_name (str): 实验名称
        
    Returns:
        Logger: 日志记录器实例
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./results/logs/{experiment_name}_{timestamp}.log"
    return Logger(log_file)
'''

with open('/home/user/utils/logger.py', 'w') as f:
    f.write(logger_code)
    
print("utils/logger.py 创建完成")