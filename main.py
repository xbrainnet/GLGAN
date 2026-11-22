import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

from config.config import Config
from utils.seed import set_seed
from utils.logger import create_experiment_logger
from data.data_loader import create_data_loaders
from training.trainer import GLGATTrainer

def run_cross_validation():
    """运行交叉验证实验"""
    # 设置随机种子
    set_seed(Config.RANDOM_SEED)
    
    # 创建日志记录器
    logger = create_experiment_logger("glgat_cv")
    
    # 打印配置
    Config.print_config()
    logger.info(f"Device: {Config.get_device_info()}")
    
    # 存储结果
    fold_results = []
    
    # K折交叉验证
    for fold in range(Config.K_FOLDS):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold + 1}/{Config.K_FOLDS}")
        logger.info(f"{'='*50}")
        
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(fold)
        
        # 创建训练器
        trainer = GLGATTrainer(device=Config.DEVICE, logger=logger)
        
        # 训练模型
        best_acc = trainer.train(train_loader, val_loader)
        
        # 最终验证
        final_metrics = trainer.validate(val_loader)
        
        fold_results.append(final_metrics)
        
        logger.info(f"Fold {fold + 1} Results:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # 计算平均结果
    avg_results = {}
    for metric in fold_results[0].keys():
        values = [result[metric] for result in fold_results]
        avg_results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # 打印最终结果
    logger.info(f"\n{'='*50}")
    logger.info("Cross-Validation Results:")
    logger.info(f"{'='*50}")
    
    for metric, stats in avg_results.items():
        logger.info(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    return avg_results

def main():
    """主函数"""
    try:
        results = run_cross_validation()
        print("\n实验完成！")
        print(f"平均准确率: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
        
    except Exception as e:
        print(f"实验过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
