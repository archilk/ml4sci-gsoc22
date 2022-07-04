import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import auroc as auroc_fn, accuracy as accuracy_fn
from vit_pytorch import ViT
from torchvision import transforms
import timm
import wandb
import numpy as np
import argparse
import os

from data import LensDataset
from constants import *
from utils import get_best_device
from networks import BaselineModel, ViTClassifier

@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    loss, accuracy, class_auroc, micro_auroc, macro_auroc = [], [], [], [], []
    logits, y = [], []
    for batch_X, batch_y in data_loader:
        batch_X, batch_y = batch_X.to(device, dtype=torch.float), batch_y.type(torch.LongTensor)
        logits.append(model(batch_X).cpu())
        y.append(batch_y)
    
    logits, y = torch.cat(logits), torch.cat(y)
    loss.append(loss_fn(logits, y))
    accuracy.append(accuracy_fn(logits, y, num_classes=NUM_CLASSES))
    class_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average=None))
    #micro_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average='micro'))
    macro_auroc.append(auroc_fn(logits, y, num_classes=NUM_CLASSES, average='macro'))

    result = {
        'ground_truth': y,
        'logits': logits,
        'loss': np.mean(loss),
        'accuracy': np.mean(accuracy),
        'micro_auroc': np.mean(micro_auroc),
        'macro_auroc': np.mean(macro_auroc)
    }

    class_auroc = class_auroc[0]
    for i, label in enumerate(LABELS):
        result[f'{label}_auroc'] = class_auroc[i]

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runid', type=str, help='ID of train run')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'best'], default='best')
    run_config = parser.parse_args()

    with wandb.init(entity='_archil', id=run_config.runid, resume='must'):
        if run_config.device == 'best':
            device = get_best_device()
        else:
            device = run_config.device

        datapath = os.path.join('./data', wandb.config.dataset, 'memmap', 'test')
        padding_transform = transforms.Compose([transforms.Pad(37)]) if wandb.config.model == 'vit_pretrained' else None
        dataset = LensDataset(memmap_path=datapath, transform=padding_transform)
        data_loader = DataLoader(dataset, batch_size=wandb.config.batchsize, shuffle=False)

        if wandb.config.model == 'baseline':
            model = BaselineModel().to(device)
        elif wandb.config.model == 'vit':
            model = ViT(image_size=IMAGE_SIZE[0], num_classes=NUM_CLASSES, channels=1,
                        patch_size=wandb.config.patch_size,
                        dim=wandb.config.projection_dim,
                        depth=wandb.config.num_transformer_layers,
                        heads=wandb.config.num_heads,
                        mlp_dim=wandb.config.mlp_dim,
                        dropout=wandb.config.dropout, emb_dropout=wandb.config.transformer_dropout).to(device)
        elif wandb.config.model == 'vit_pretrained':
            model = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=1, num_classes=NUM_CLASSES).to(device)
        else:
            model = None
        weights_file = wandb.restore('best_model.pt')
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'best_model.pt')))

        criterion = CrossEntropyLoss()

        metrics = evaluate(model, data_loader, criterion, device=device)

        wandb.run.summary['test_loss'] = metrics['loss']
        wandb.run.summary['test_accuracy'] = metrics['accuracy']
        wandb.run.summary['test_micro_auroc'] = metrics['micro_auroc']
        wandb.run.summary['test_macro_auroc'] = metrics['macro_auroc']
        for label in LABELS:
            wandb.run.summary[f'test_{label}_auroc'] = metrics[f'{label}_auroc']

        wandb.log({
            'test_roc': wandb.plot.roc_curve(metrics['ground_truth'],
                                        torch.nn.functional.softmax(metrics['logits']),
                                        labels=LABELS)
        })

