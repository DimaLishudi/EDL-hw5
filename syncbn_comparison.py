import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100

from syncbn import SyncBatchNorm
import argparse
from time import perf_counter

torch.set_num_threads(1)


def init_process(local_rank, bn="no"):
    """Initialize the distributed environment."""
    dist.init_process_group("gloo", rank=local_rank)
    size = dist.get_world_size()
    run_experiments(local_rank, bn)


# added scale: we need to divide by grad_accum_steps
# otherwise for each batch size we need to choose new step size
def average_gradients(model, scale=1):
    size = float(dist.get_world_size()) * scale
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def get_atol_rtol_time(local_rank, batch_size, n_features, bn):
    torch.manual_seed(0)
    custom_bn = SyncBatchNorm(n_features)
    reference_bn = nn.SyncBatchNorm(n_features, affine=False) if bn == "sync" else nn.BatchNorm1d(n_features, affine=False)
    torch.manual_seed(local_rank)
    data = torch.randn(batch_size//2, n_features)

    custom_forward = custom_bn(data)
    reference_forward = reference_bn(data)
    custom_backward = custom_forward.sum().backward()
    reference_backward = reference_forward.sum().backward()



def run_experiments(local_rank, bn):
    for batch_size in [32, 64]:
        for n_features in [128, 256, 512, 1024]:
            atol, rtol, time = get_atol_rtol_time(local_rank, batch_size, n_features, bn)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, nargs='?', default="cpu")
    parser.add_argument('-b', '--batchnorm', type=str, nargs='?', default="sync")
    args = parser.parse_args()
    init_process(local_rank, fn=run_training, bn=args.batchnorm)
