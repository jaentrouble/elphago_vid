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
        acc1_avg = tm.Accuracy(task='multiclass', num_classes=44).to('cuda')
        acc2_avg = tm.Accuracy(task='multiclass', num_classes=44).to('cuda')
        if self.epochs < 0:
            t = tqdm.tqdm(iter(itertools.count()), ncols=100, unit='epoch')
        else:
            t = tqdm.trange(self.epochs, ncols=100)
        best_acc = 0
        best_epoch = 0
        for epoch in t:
            t_epoch = tqdm.tqdm(self.train_loader, ncols=100, leave=False)
            last_n = 0
            for x, y1, y2 in t_epoch:
                x = x.to('cuda')
                y1 = y1.to('cuda')
                y2 = y2.to('cuda')
                logits = self.model(x)
                logits = logits.reshape(logits.shape[0], 2, -1)
                logit1 = logits[:, 0, :]
                logit2 = logits[:, 1, :]
                loss1 = self.loss_fn(logit1, y1)
                loss2 = self.loss_fn(logit2, y2)
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_avg.update(loss)
                acc1_avg.update(logit1, y1)
                acc2_avg.update(logit2, y2)

                if t_epoch.n - last_n > 100:
                    t_epoch.set_postfix(loss=loss_avg.compute().item(), 
                                        acc1=acc1_avg.compute().item(),
                                        acc2=acc2_avg.compute().item())

                    last_n = t_epoch.n
            self.tb_writer.add_scalar('loss', loss_avg.compute(), epoch)
            self.tb_writer.add_scalar('acc1', acc1_avg.compute(), epoch)
            self.tb_writer.add_scalar('acc2', acc2_avg.compute(), epoch)
            acc_avg = (acc1_avg.compute() + acc2_avg.compute()) / 2
            self.lr_scheduler.step(acc_avg)
            if acc_avg > best_acc:
                best_acc = acc_avg
                best_epoch = epoch
                torch.save(self.model.state_dict(), self.log_dir/f'checkpoints/best_acc.pt')
            else:
                if epoch - best_epoch > self.patience:
                    break
            t.set_postfix(loss=loss_avg.compute().item(), acc=acc_avg.item(),
                          patience=epoch-best_epoch, lr=self.optimizer.param_groups[0]['lr'])
            loss_avg.reset()
            acc1_avg.reset()
            acc2_avg.reset()
        t.close()

if __name__ == '__main__':
    with open('configs/config_advice_two.json') as f:
        config = json.load(f)
    trainer = Trainer(**config)
    with open(trainer.log_dir/'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    trainer.train()