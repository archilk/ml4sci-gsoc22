import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchmetrics.functional import auroc as auroc_fn, accuracy as accuracy_fn
import wandb
import numpy as np
import argparse
import os

from data import LensDataset
from constants import *
from utils import get_best_device
from networks import BaselineModel

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
    parser.add_argument('--dataset', choices=['Model_I', 'Model_II', 'Model_III', 'Model_IV'], default='I', help='which data model')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--optim', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--model', type=str)
    parser.add_argument('--runname', type=str, help='Name of run on wandb')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'best'], default='best')
    run_config = parser.parse_args()

    with wandb.init(entity='_archil', config=run_config, group=f'{run_config.dataset}', job_type='test', id=wandb.util.generate_id()):
        if run_config.device == 'best':
            device = get_best_device()
        else:
            device = run_config.device

        datapath = os.path.join('./data', f'{run_config.dataset}', 'memmap', 'test')
        dataset = LensDataset(memmap_path=datapath)
        data_loader = DataLoader(dataset, batch_size=run_config.batch_size, shuffle=False)

        model = BaselineModel().to(device)
        weights_file = wandb.restore('best_model.pt', run_path=f'_archil/{run_config.model}/{run_config.runname}')
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, 'best_model.pt')))

        criterion = CrossEntropyLoss()

        metrics = evaluate(model, data_loader, criterion, device=device)

        log_dict = {'test/loss': metrics['loss'], 'test/accuracy': metrics['accuracy'],
                   'test/micro_auroc': metrics['micro_auroc'], 'test/macro_auroc': metrics['macro_auroc']}
        for label in LABELS:
            log_dict[f'test/{label}_auroc'] = metrics[f'{label}_auroc']
        wandb.log(log_dict)

        wandb.log({
            'roc': wandb.plot.roc_curve(metrics['ground_truth'],
                                        torch.nn.functional.softmax(metrics['logits']),
                                        labels=LABELS)
        })

