import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import models
from pathlib import Path
import json
import datasets
import torchmetrics as tm
import tqdm

class Trainer():
    def __init__(
        self,
        config:dict,
    ) -> None:
        self.test_name = config['test_name']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.lr = config['lr']
        self.memo = config['memo']

        self.log_dir = Path('logs')/self.test_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir/'checkpoints').mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
        with open(self.log_dir/'config.json', 'w') as f:
            json.dump(config, f, indent=4)

        self.model = getattr(models, config['model_name'])(**config['model_kwargs'])
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ds = getattr(datasets, config['dataset_name'])(**config['dataset_kwargs'])
        self.train_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def train(self):
        loss_avg = tm.MeanMetric()
        acc_avg = tm.Accuracy(task='multiclass', num_classes=14)
        t = tqdm.trange(self.epochs, ncols=100)
        for epoch in t:
            for x, y in self.train_loader:
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                loss_avg.update(loss)
                acc_avg.update(y_hat.argmax(dim=1), y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.tb_writer.add_scalar('loss', loss_avg.compute(), epoch)
            self.tb_writer.add_scalar('acc', acc_avg.compute(), epoch)
            t.set_postfix(loss=loss_avg.compute().item(), acc=acc_avg.compute().item())
            loss_avg.reset()
            acc_avg.reset()
            torch.save(self.model.state_dict(), self.log_dir/f'checkpoints/epoch_{epoch}.pt')

if __name__ == '__main__':
    with open('configs/config_enchant_n.json') as f:
        config = json.load(f)
    trainer = Trainer(config)
    trainer.train()