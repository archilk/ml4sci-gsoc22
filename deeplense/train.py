import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from vit_pytorch import ViT
from tqdm import tqdm
import numpy as np
import argparse
import os
import wandb
import timm

from data import LensDataset, WrapperDataset
from constants import *
from utils import get_best_device, set_seed
from networks import BaselineModel, ViTClassifier, ViTPretrainedClassifier
from eval import evaluate

def train_step(model, images, labels, optimizer, scheduler, criterion, device='cpu'):
    images, labels = images.to(device, dtype=torch.float), labels.type(torch.LongTensor).to(device)
    model.train()
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step(loss)
    return loss


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, log_interval=100):
    wandb.watch(model, criterion, log='all', log_freq=log_interval)
    best_val_auroc, best_val_metrics = 0., dict()
    batch_num = 0
    for epoch in range(1, epochs + 1):
        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch_num += 1
            model.train()
            images, labels = batch_data
            loss = train_step(model, images, labels, optimizer, scheduler, criterion, device=device)

            if batch_num % log_interval == 0:
                val_metrics = evaluate(model, val_loader, criterion, device=device)

                # Log in wandb
                log_dict = {
                    'epoch': epoch,
                    'batch_num': batch_num,
                    'train/loss': loss,
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/micro_auroc': val_metrics['micro_auroc'],
                    'val/macro_auroc': val_metrics['macro_auroc']
                }
                for label in LABELS:
                    log_dict[f'val/{label}_auroc'] = val_metrics[f'{label}_auroc']
                wandb.log(log_dict, step=batch_num)
                wandb.log({
                    'roc': wandb.plot.roc_curve(val_metrics['ground_truth'],
                                                torch.nn.functional.softmax(val_metrics['logits']),
                                                labels=LABELS)
                })

                # Track best val auroc
                if val_metrics['macro_auroc'] > best_val_auroc:
                    best_val_auroc = val_metrics['macro_auroc']
                    wandb.run.summary['best_val_micro_auroc'] = val_metrics['micro_auroc']
                    wandb.run.summary['best_val_macro_auroc'] = best_val_auroc
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_batch_num'] = batch_num
                    for label in LABELS:
                        wandb.run.summary[f'best_val_{label}_auroc'] = val_metrics[f'{label}_auroc']
                    best_val_metrics = val_metrics
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'best_model.pt'))

        # Sync best model at a lesser frequency (i.e. at the end of each epoch)
        wandb.save(os.path.join(wandb.run.dir, 'best_model.pt'))

    return best_val_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Fixed params
    parser.add_argument('--dataset', choices=['Model_I', 'Model_II', 'Model_III', 'Model_IV'], default='Model_I', help='which data model')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--model', choices=['baseline', 'vit', 'vit_pretrained'], default='vit')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'tpu', 'best'], default='best')
    parser.add_argument('--random_zoom', type=float, default=0.8)
    parser.add_argument('--random_rotation', type=float, default=180)

    # Common hyperparameters
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--decay_lr', type=int, default=0)

    # ViT hyperparameters
    parser.add_argument('--patch_size', type=int, default=30)
    parser.add_argument('--projection_dim', type=int, default=1024)
    parser.add_argument('--num_transformer_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--mlp_dim', type=int, default=2048)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--tune', type=int, choices=[0, 1], default=1, help='Whether to further tune (1) pretrained model (if any) or freeze the pretrained weights (0)')

    run_config = parser.parse_args()

    tune = True if run_config.tune == 1 else False

    with wandb.init(entity='_archil', config=run_config, group=f'{run_config.dataset}', job_type='train'):
        if run_config.seed:
            set_seed(run_config.seed)

        if run_config.device == 'best':
            device = get_best_device()
        elif run_config.device == 'tpu':
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        else:
            device = run_config.device
        
        if run_config.dataset == 'Model_I':
            IMAGE_SIZE = 150
        elif run_config.dataset == 'Model_II' or run_config.dataset == 'Model_III':
            IMAGE_SIZE = 64
        else:
            IMAGE_SIZE = None

        datapath = os.path.join('./data', f'{run_config.dataset}', 'memmap', 'train')
        train_dataset = LensDataset(image_size=IMAGE_SIZE, memmap_path=datapath)
        # 90%-10% Train-validation split
        train_size = int(len(train_dataset) * 0.8)
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        augment_transforms = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        if run_config.random_rotation > 0:
            augment_transforms.append(transforms.RandomRotation(run_config.random_rotation, interpolation=transforms.InterpolationMode.BILINEAR, fill=-1.))
        if run_config.random_zoom < 1: # 1 is when the random crop is the whole image
            augment_transforms.append(transforms.RandomResizedCrop(IMAGE_SIZE, scale=(run_config.random_zoom**2, 1.), ratio=(1., 1.)))
        padding_transform = [transforms.Pad(37, fill=-1.)] if run_config.model == 'vit_pretrained' else None
        if padding_transform is not None:
            augment_transforms.extend(padding_transform) # 150x150 to 224x224 for ViT
            padding_transform = transforms.Compose(padding_transform)
        augment_transforms = transforms.Compose(augment_transforms)

        train_dataset = WrapperDataset(train_dataset, transform=augment_transforms)
        val_dataset = WrapperDataset(val_dataset, transform=padding_transform)

        train_loader = DataLoader(train_dataset, batch_size=run_config.batchsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=run_config.batchsize, shuffle=False)

        if run_config.model == 'baseline':
            model = BaselineModel(image_size=IMAGE_SIZE, dropout_rate=run_config.dropout).to(device)
        elif run_config.model == 'vit':
            model = ViT(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES, channels=1,
                        patch_size=run_config.patch_size,
                        dim=run_config.projection_dim,
                        depth=run_config.num_transformer_layers,
                        heads=run_config.num_heads,
                        mlp_dim=run_config.mlp_dim,
                        dropout=run_config.dropout, emb_dropout=run_config.transformer_dropout).to(device)
        elif run_config.model == 'vit_pretrained':
            model = ViTPretrainedClassifier(dropout_rate=run_config.dropout, tune=tune).to(device)
        else:
            model = None

        if run_config.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=run_config.lr)
        else:
            optimizer = SGD(model.parameters(), lr=run_config.lr)
        
        criterion = CrossEntropyLoss()

        if run_config.decay_lr == 1:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6, verbose=True)
        else:
            scheduler = None

        best_val_metrics = train(model, train_loader, val_loader, criterion, optimizer, scheduler, run_config.epochs,
                                 device, run_config.log_interval)

