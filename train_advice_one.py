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
import itertools

class Trainer():
    def __init__(
            self,
            test_name:str,
            batch_size:int,
            epochs:int,
            patience:int,
            lr:float,
            memo:str,
            model_name:str,
            model_kwargs:dict,
            dataset_name:str,
            dataset_kwargs:dict,
    ) -> None:
        self.test_name = test_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.memo = memo

        self.log_dir = Path('logs')/self.test_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir/'checkpoints').mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))

        self.model = getattr(models, model_name)(**model_kwargs).to('cuda')

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=patience//2, 
        )

        ds = getattr(datasets, dataset_name)(**dataset_kwargs)
        self.train_loader = DataLoader(
            ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=16,
        )
    
    def train(self):
        loss_avg = tm.MeanMetric().to('cuda')
        acc_avg = tm.Accuracy(task='multiclass', num_classes=132).to('cuda')
        if self.epochs < 0:
            t = tqdm.tqdm(iter(itertools.count()), ncols=100, unit='epoch')
        else:
            t = tqdm.trange(self.epochs, ncols=100)
        best_acc = 0
        best_epoch = 0
        for epoch in t:
            t_epoch = tqdm.tqdm(self.train_loader, ncols=100, leave=False)
            last_n = 0
            for x, y in t_epoch:
                x = x.to('cuda')
                y = y.to('cuda')
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                loss_avg.update(loss)
                acc_avg.update(y_hat.argmax(dim=1), y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if t_epoch.n - last_n > 100:
                    t_epoch.set_postfix(loss=loss_avg.compute().item(), 
                                        acc=acc_avg.compute().item())
                    last_n = t_epoch.n
            self.tb_writer.add_scalar('loss', loss_avg.compute(), epoch)
            self.tb_writer.add_scalar('acc', acc_avg.compute(), epoch)
            self.lr_scheduler.step(acc_avg.compute())
            if acc_avg.compute() > best_acc:
                best_acc = acc_avg.compute()
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.log_dir/f'checkpoints/best_acc.pt')
            else:
                if epoch - best_epoch > self.patience:
                    break
            t.set_postfix(loss=loss_avg.compute().item(), acc=acc_avg.compute().item(),
                          patience=epoch-best_epoch, lr=self.optimizer.param_groups[0]['lr'])
            loss_avg.reset()
            acc_avg.reset()
        t.close()

if __name__ == '__main__':
    with open('configs/config_advice_one.json') as f:
        config = json.load(f)
    trainer = Trainer(**config)
    with open(trainer.log_dir/'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    trainer.train()