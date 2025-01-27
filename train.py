import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 测试函数
def test(model, datasets_test):
    eval_loss = 0
    eval_acc = 0
    pre_all = []
    labels_all = []
    pro_all = []
    model.eval()
    for node, edge, label in datasets_test:
        node, edge, label = node.to(DEVICE), edge.to(DEVICE), label.to(DEVICE)
        node = node.float()
        edge = edge.float()
        label = label.long()
        out, loss_contrast = model(node, edge)
        loss_class = F.nll_loss(out, label)
        loss = loss_class + beta * loss_contrast
        eval_loss += float(loss)
        _, pred = out.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / node.shape[0]
        eval_acc += acc
        pre = pred.cpu().detach().numpy()
        pre_all.extend(pre)
        label_true = label.cpu().detach().numpy()
        labels_all.extend(label_true)
        pro_all.extend(out[:, 1].cpu().detach().numpy())
    tn, fp, fn, tp = confusion_matrix(labels_all, pre_all).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    eval_acc_epoch = accuracy_score(labels_all, pre_all)
    my_auc = roc_auc_score(labels_all, pro_all)
    precision = precision_score(labels_all, pre_all)
    recall = recall_score(labels_all, pre_all)
    f1 = f1_score(labels_all, pre_all)

    return eval_loss, eval_acc, eval_acc_epoch, specificity, sensitivity, f1, my_auc, labels_all, pro_all