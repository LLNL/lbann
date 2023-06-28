import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.cuda.amp import autocast, GradScaler


class Trainer:
    def __init__(self, model, optimizer, train_dataloader, eval_dataloader, num_epochs, device, enable_amp=False, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.num_epochs = num_epochs
        self.device = device
        self.epoch = 0
        self.rank = dist.get_rank()
        self.enable_amp = enable_amp
        self.scaler = GradScaler(enabled=self.enable_amp)
        self.scheduler = scheduler
    
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        with autocast(enabled=self.enable_amp):
            loss = F.mse_loss(self.model(x), y)
        self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train_epoch(self):
        self.model.train()

        iter = self.train_dataloader
        if self.rank == 0:
            iter = tqdm(self.train_dataloader, desc=f'Epoch {self.epoch}')
        for x, y in iter:
            self.train_step(x.to(self.device), y.to(self.device))
        if self.scheduler is not None:
            self.scheduler.step()
    
    def train(self):
        for self.epoch in range(self.num_epochs):
            self.train_epoch()
            mse = self.eval()
            if self.rank == 0:
                print(f'Epoch {self.epoch} Validation MSE: {mse}')
    
    @torch.no_grad()
    def eval(self):
        self.model.eval()

        score = 0
        count = 0
        with autocast(enabled=self.enable_amp):
            for x, y in self.eval_dataloader:
                score += (self.model(x.to(self.device)) - y.to(self.device)).square().mean(dim=-1).sum()
                count += len(y)
        count = torch.tensor(count).to(self.device)
        
        dist.all_reduce(score, ReduceOp.SUM)
        dist.all_reduce(count, ReduceOp.SUM)
        score /= count

        return score.item()
