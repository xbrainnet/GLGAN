import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from models.glgat import GLGAT
from config.config import Config
from utils.logger import Logger

class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    
    def __init__(self, temperature=Config.TAU):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        计算对比损失
        
        Args:
            features: 归一化特征 [batch_size, feature_dim]
            labels: 标签 [batch_size]
        """
        batch_size = features.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 移除对角线
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # 计算损失
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
        
        log_prob = similarity_matrix - torch.log(sum_exp_sim)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        loss = -mean_log_prob_pos.mean()
        return loss

class GLGATTrainer:
    """GLGAT训练器"""
    
    def __init__(self, device=None, logger=None):
        self.device = device or Config.DEVICE
        self.logger = logger or Logger()
        
        # 初始化模型
        self.model = GLGAT().to(self.device)
        
        # 损失函数
        self.classification_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE
        )
        
        # 早停
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        predictions = []
        true_labels = []
        
        for batch in train_loader:
            fmri_data = batch['fmri'].to(self.device)
            dti_data = batch['dti'].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits, contrastive_features = self.model(fmri_data, dti_data)
            
            # 计算损失
            cls_loss = self.classification_loss(logits, labels)
            cont_loss = self.contrastive_loss(contrastive_features, labels)
            total_loss_batch = cls_loss + Config.BETA * cont_loss
            
            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # 记录预测结果
            pred = torch.argmax(logits, dim=1)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        
        return total_loss / len(train_loader), accuracy
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        true_labels = []
        probabilities = []
        
        with torch.no_grad():
            for batch in val_loader:
                fmri_data = batch['fmri'].to(self.device)
                dti_data = batch['dti'].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                
                # 前向传播
                logits, contrastive_features = self.model(fmri_data, dti_data)
                
                # 计算损失
                cls_loss = self.classification_loss(logits, labels)
                cont_loss = self.contrastive_loss(contrastive_features, labels)
                total_loss_batch = cls_loss + Config.BETA * cont_loss
                
                total_loss += total_loss_batch.item()
                
                # 记录结果
                pred = torch.argmax(logits, dim=1)
                prob = torch.softmax(logits, dim=1)
                
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                probabilities.extend(prob.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # AUC (如果是二分类)
        if len(np.unique(true_labels)) == 2:
            auc = roc_auc_score(true_labels, np.array(probabilities)[:, 1])
        else:
            auc = 0.0
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """完整训练过程"""
        self.logger.info("开始训练...")
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 日志记录
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # 早停检查
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_model('best_model.pth')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= Config.PATIENCE:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return self.best_val_acc
    
    def save_model(self, filename):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc
        }, Config.MODEL_SAVE_PATH + filename)
    
    def load_model(self, filename):
        """加载模型"""
        checkpoint = torch.load(Config.MODEL_SAVE_PATH + filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
