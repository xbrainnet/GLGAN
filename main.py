from model import GLGAT
from data_process import Dianxian
from train import test
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold

# Hyperparameters
dropout = 0.5
alpha = 0.2
beta = 1e7
epochs = 300
batch_sizes = [(20, 13)]  # (train_batch_size, test_batch_size)
learning_rates = [2e-5]

# Dataset
dataset = Dianxian()

# Cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=7)
for bs_train, bs_test in batch_sizes:
    for lr in learning_rates:
        for train_idx, test_idx in kf.split(dataset):
            train_subsampler = SubsetRandomSampler(train_idx)
            test_subsampler = SubsetRandomSampler(test_idx)
            
            datasets_train = DataLoader(dataset, batch_size=bs_train, shuffle=False, sampler=train_subsampler)
            datasets_test = DataLoader(dataset, batch_size=bs_test, shuffle=False, sampler=test_subsampler)
            
            model = GLGAT(90, 90, dropout=dropout, alpha=alpha)
            model.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                train_acc = 0
                for node, edge, label in datasets_train:
                    node, edge, label = node.to(DEVICE), edge.to(DEVICE), label.to(DEVICE)
                    node = node.float()
                    edge = edge.float()
                    label = label.long()
                    out, loss_contrast = model(node, edge)
                    loss_class = F.nll_loss(out, label)
                    loss = loss_class + beta * loss_contrast
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += float(loss)
                    _, pred = out.max(1)
                    num_correct = (pred == label).sum()
                    acc = int(num_correct) / node.shape[0]
                    train_acc += acc
                
                # Evaluate on test set
                eval_loss, eval_acc, eval_acc_epoch, precision, recall, f1, auc, labels_all, pro_all = test(model, datasets_test)
                print(f"Epoch {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_acc}, Eval Loss: {eval_loss}, Eval Accuracy: {eval_acc_epoch}")