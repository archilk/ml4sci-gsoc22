import torch
from torch.nn import Softmax
import numpy as np
import matplotlib.pyplot as plt
import os

from constants import *

@torch.no_grad()
def plot_roc_curve(model, data_loader, save_path, device, num_thresholds=200):
    model.eval()
    y_pred, y_true = [], []

    softmax = Softmax(dim=1)

    for X, y in data_loader:
        X = X.to(device, dtype=torch.float)
        pred = softmax(model(X)).cpu()
        
        y_true.append(y)
        y_pred.append(pred)
    
    y_pred, y_true = torch.cat(y_pred).numpy(), torch.cat(y_true).numpy()

    fig, axis = plt.subplots(figsize=(15, 5))
    y_onehot = np.eye(NUM_CLASSES)
    y_onehot = y_onehot[y_true]
    thresholds = np.linspace(0., 1., num_thresholds)

    tp, fp, tn, fn = [], [], [], []

    for threshold in thresholds:
        y_pred_label = (y_pred > threshold).astype(int)
        tp.append(np.count_nonzero(((y_pred_label == 1) & (y_onehot == 1)), axis=0))
        fp.append(np.count_nonzero(((y_pred_label == 1) & (y_onehot == 0)), axis=0))
        tn.append(np.count_nonzero(((y_pred_label == 0) & (y_onehot == 0)), axis=0))
        fn.append(np.count_nonzero(((y_pred_label == 0) & (y_onehot == 1)), axis=0))

    tp, fp, tn, fn = np.array(tp), np.array(fp), np.array(tn), np.array(fn)

    tp_rate = tp / (tp + fn)
    fp_rate = fp / (fp + tn)

    for label, label_idx in LABEL_MAP.items():
        _ = axis.plot(fp_rate[:, label_idx], tp_rate[:, label_idx], label=label)
    
    _ = fig.savefig(save_path)


def get_best_device():
    device = 'cpu'
    if torch.has_mps:
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

