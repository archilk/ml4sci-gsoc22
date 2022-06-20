import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import auroc as auroc_fn, accuracy as accuracy_fn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os

from data import LensDataset
from constants import *
from utils import plot_roc_curve, get_best_device
from networks import BaselineModel

@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    loss, accuracy, auc = [], [], []
    logits, y = [], []
    for batch_X, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device, dtype=torch.float), batch_y.type(torch.LongTensor)
        logits.append(model(batch_X).cpu())
        y.append(batch_y)
    
    logits, y = torch.cat(logits), torch.cat(y)
    loss.append(loss_fn(logits, y))
    accuracy.append(accuracy_fn(logits, y, num_classes=NUM_CLASSES))
    auc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES))
    
    return {
        'loss': np.mean(loss),
        'accuracy': np.mean(accuracy),
        'auc': np.mean(auc)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/Model_I/memmap/test', help='root directory for dataset')
    parser.add_argument('--model-path', type=str, default='./checkpoints/baseline/best_model.pt')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model', type=str)
    parser.add_argument('--save-dir', type=str, default='./checkpoints/baseline')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'best'], default='best')
    args = parser.parse_args()

    if args.device == 'best':
        device = get_best_device()
    else:
        device = args.device

    dataset = LensDataset(memmap_path=args.data_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = BaselineModel().to(device)
    model.load_state_dict(torch.load(args.model_path))

    loss_fn = CrossEntropyLoss()

    writer = SummaryWriter(os.path.join(args.save_dir, 'tb_logs'))
    metrics = evaluate(model, data_loader, loss_fn, device)
    writer.add_scalars('loss', {'test': metrics['loss']})
    writer.add_scalars('accuracy', {'test': metrics['accuracy']})
    writer.add_scalars('auc', {'test': metrics['auc']})
    writer.flush()

    plot_roc_curve(model, data_loader, os.path.join(args.save_dir, 'test_roc.jpg'), device)

    writer.close()

