import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import argparse
import os
import wandb

from data import LensDataset, WrapperDataset
from constants import *
from utils import get_best_device, set_seed
from networks import BaselineModel
from eval import evaluate

def train_step(model, images, labels, optimizer, criterion, device='cpu'):
    images, labels = images.to(device, dtype=torch.float), labels.type(torch.LongTensor).to(device)
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    return loss


def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, log_interval=100):
    wandb.watch(model, criterion, log='all', log_freq=log_interval)
    best_val_auroc, best_val_metrics = 0., dict()
    batch_num = 0
    for epoch in range(1, epochs + 1):
        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch_num += 1
            model.train()
            images, labels = batch_data
            loss = train_step(model, images, labels, optimizer, criterion, device=device)

            if batch_num % log_interval == 0:
                val_metrics = evaluate(model, val_loader, criterion, device=device)

                # Log in wandb
                log_dict = {
                    'epoch': epoch,
                    'train/loss': loss,
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/micro_auroc': val_metrics['micro_auroc'],
                    'val/macro_auroc': val_metrics['macro_auroc']
                }
                for label in LABELS:
                    log_dict[f'val/{label}_auroc'] = val_metrics[f'{label}_auroc']
                wandb.log(log_dict, step=batch_num)

                # Track best val auroc
                if val_metrics['macro_auroc'] > best_val_auroc:
                    best_val_auroc = val_metrics['macro_auroc']
                    wandb.run.summary['best_val_auroc'] = best_val_auroc
                    best_val_metrics = val_metrics
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))

        # Sync best model at a lesser frequency (i.e. at the end of each epoch)
        wandb.save(os.path.join(wandb.run.dir, 'best_model.pt'))

    return best_val_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['Model_I', 'Model_II', 'Model_III', 'Model_IV'], default='Model_I', help='which data model')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--model', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'best'], default='best')
    run_config = parser.parse_args()

    with wandb.init(entity='_archil', config=run_config, group=f'{run_config.dataset}', job_type='train'):
        if run_config.seed:
            set_seed(run_config.seed)

        if run_config.device == 'best':
            device = get_best_device()
        else:
            device = run_config.device

        datapath = os.path.join('./data', f'{run_config.dataset}', 'memmap', 'train')
        train_dataset = LensDataset(memmap_path=datapath)
        # 90%-10% Train-validation split
        train_size = int(len(train_dataset) * 0.8)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        augment_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                transforms.RandomVerticalFlip(),
                                                transforms.RandomResizedCrop(150, scale=(0.8, 1)),
                                                transforms.RandomRotation(10)])
        train_dataset = WrapperDataset(train_dataset, transform=augment_transforms)
        val_dataset = WrapperDataset(val_dataset)

        train_loader = DataLoader(train_dataset, batch_size=run_config.batchsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=run_config.batchsize, shuffle=False)

        model = BaselineModel(dropout_rate=run_config.dropout).to(device)

        if run_config.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=run_config.lr)
        else:
            optimizer = SGD(model.parameters(), lr=run_config.lr)
        
        criterion = CrossEntropyLoss()

        best_val_metrics = train(model, train_loader, val_loader, criterion, optimizer, run_config.epochs,
                                 device, run_config.log_interval)

        wandb.log({
            'roc': wandb.plot.roc_curve(best_val_metrics['ground_truth'],
                                        torch.nn.functional.softmax(best_val_metrics['logits']),
                                        labels=LABELS)
        })
