import torch
from torch.utils.data import DataLoader, DistributedSampler
from data import CosmoflowDataset
from trainer import Trainer
from model import CosmoFlow
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse


parser = argparse.ArgumentParser(prog='CosmoFlow')

parser.add_argument('--learning-rate', default=1e-3, type=float,
                    help='Learning Rate (default: 1e-3)')
parser.add_argument('--batch-size', default=8, type=int,
                    help='Batch Size (default: 8)')
parser.add_argument('--num-epochs', default=50, type=int,
                    help='Number of Training Epochs (default: 50)')
parser.add_argument('--input-width', default=128, type=int,
                    help='The spatial size of the input cube (default: 128)')
parser.add_argument('--train-data',
                    default='/p/vast1/lbann/datasets/cosmoflow/128_small',
                    type=str,
                    help='Training Data Directory')
parser.add_argument('--val-data',
                    default='/p/vast1/lbann/datasets/cosmoflow/128_small',
                    type=str,
                    help='Validation Data Directory')
parser.add_argument('--preload', action='store_true',
                    help='Preload Training/Validation Data')
parser.add_argument('--use-batchnorm', action='store_true',
                    help='Use batch normalization layers')
parser.add_argument('--enable-amp', action='store_true',
                    help='Use automatic mixed precision')

args = parser.parse_args()

dist.init_process_group("nccl", init_method="env://",
    world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']),
    rank=int(os.environ['OMPI_COMM_WORLD_RANK']))

rank = dist.get_rank()
device = rank % torch.cuda.device_count()

train_dataset = CosmoflowDataset(args.train_data, args.preload)
eval_dataset = CosmoflowDataset(args.val_data, args.preload)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

model = CosmoFlow(args.input_width, args.use_batchnorm).to(device)
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[device])
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)

def lr_lambda(epoch):
    if epoch < 4:
        return 1e-2 + (1 - 1e-2) * epoch / 4
    elif epoch >= 64:
        return 0.125
    elif epoch >= 32:
        return 0.25
    else:
        return 1
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

trainer = Trainer(model, optimizer, train_dataloader, eval_dataloader, args.num_epochs, device, args.enable_amp, scheduler)
trainer.train()