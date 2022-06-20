import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchmetrics.functional import auroc as auroc_fn, accuracy as accuracy_fn
from tqdm import tqdm
import numpy as np
import argparse
import os
import random

from data import LensDataset
from constants import *
from utils import plot_roc_curve, get_best_device
from networks import BaselineModel
from eval import evaluate

def train(model, train_loader, val_loader, loss_fn, optimizer, epochs, save_path, device):
    train_accuracy, val_accuracy = [], []
    train_loss, val_loss = [], []
    train_auc, val_auc = [], []

    temp_loss, temp_accuracy, temp_auc = [], [], []

    best_val_auc = 0.

    batch_num = 0
    for epoch in range(1, epochs + 1):
        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch_num += 1
            model.train()
            X, y = batch_data
            X, y = X.to(device, dtype=torch.float), y.type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            temp_loss.append(loss.detach().cpu())
            temp_accuracy.append(accuracy_fn(logits, y, num_classes=NUM_CLASSES).cpu())
            temp_auc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES).cpu())

            if batch_num % 10 == 0:
                train_loss.append(np.mean(temp_loss))
                train_accuracy.append(np.mean(temp_accuracy))
                train_auc.append(np.mean(temp_auc))
                temp_loss, temp_accuracy = [], []

                metrics = evaluate(model, val_loader, loss_fn, device)
                val_loss.append(metrics['loss'])
                val_accuracy.append(metrics['accuracy'])
                val_auc.append(metrics['auc'])

                if metrics['auc'] > best_val_auc:
                    best_val_auc = metrics['auc']
                    torch.save(model.state_dict(), save_path)
    
    return {
        'train': {'loss': train_loss, 'accuracy': train_accuracy, 'auc': train_auc},
        'val': {'loss': val_loss, 'accuracy': val_accuracy, 'auc': val_auc}
    }


if __name__ == '__main__':
    # TODO: Add validation for filepath
    # TODO: Add Tensorboard logs
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data/Model_I/memmap/train', help='root directory for dataset')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--optim', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--model', type=str)
    parser.add_argument('--save-dir', type=str, default='./checkpoints/baseline')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'best'], default='best')
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.device == 'best':
        device = get_best_device()
    else:
        device = args.device

    augment_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.RandomResizedCrop(150, scale=(0.8, 1)),
                                            transforms.RandomRotation(10)])
    train_dataset = LensDataset(memmap_path=args.data_dir, transform=augment_transforms)
    # 90%-10% Train-validation split
    train_size = int(len(train_dataset) * 0.9)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = BaselineModel(dropout_rate=parser.dropout).to(device)

    if args.optim == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr)
    else:
        optimizer = None
    
    loss_fn = CrossEntropyLoss()

    model_save_path = os.path.join(args.save_dir, 'best_model.pt')
    history = train(model, train_loader, val_loader, loss_fn, optimizer, args.epochs, model_save_path, device)
    
    model.load_state_dict(torch.load(model_save_path))

    train_metrics = evaluate(model, train_loader, loss_fn, device)
    val_metrics = evaluate(model, val_loader, loss_fn, device)

    plot_roc_curve(model, val_loader, os.path.join(args.save_dir, 'val_roc.jpg'), device)

