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

torch.set_num_threads(1)


def init_process(local_rank, fn, device_type="cpu"):
    """Initialize the distributed environment."""
    backend = "gloo" if device_type == "cpu" else "nccl"
    device = "cpu" if device_type == "cpu" else torch.device("cuda", local_rank)
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size, device)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        self.bn1 = SyncBatchNorm(128, affine=False)  # to be replaced with SyncBatchNorm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

# added scale: we need to divide by grad_accum_steps
# otherwise for each batch size we need to choose new step size
def average_gradients(model, scale=1):
    size = float(dist.get_world_size()) * scale
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


# I know that we were tasked to use dist.scatter
# but it is terribly inefficient here, as all processes must wait for one scatter
def get_dataset_shard(dataset, local_rank):
    data_size = len(dataset)
    world_size = dist.get_world_size()
    shard_size = len(range(0, data_size, world_size))
    pic_shape = dataset[0].shape

    shard = torch.empty(shard_size, *pic_shape)
    targets = torch.empty(shard_size)
    for i, j in enumerate(range(0, data_size, world_size)):
        shard[i], targets[i] = dataset[j]
    return shard, targets


def run_training(rank, size, device):
    torch.manual_seed(0)
    world_size = dist.get_world_size()

    dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
    )
    valid_dataset = CIFAR100(
        "./cifar",
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
        train=False
    )
    shard, targets = get_dataset_shard(valid_dataset, rank)
    batch_size=64
    loader = DataLoader(dataset, sampler=DistributedSampler(dataset, size, rank), batch_size=batch_size)
    valid_loader = DataLoader(TensorDataset(shard, targets), batch_size=batch_size)
    grad_accum_steps = 2

    model = Net()
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    num_batches = len(loader)

    optimizer.zero_grad()
    batch_count = 0
    for _ in range(10):
        epoch_loss_acc = torch.zeros((2,), device=device)

        for data, target in loader:
            batch_count += 1
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            epoch_loss_acc[0] += loss.detach() / num_batches
            loss.backward()
            if batch_count % grad_accum_steps == 0:
                average_gradients(model, grad_accum_steps)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss_acc[1] += (output.argmax(dim=1) == target).float().mean() / num_batches

        dist.all_reduce(epoch_loss_acc.data, op=dist.ReduceOp.SUM)
        epoch_loss_acc /= world_size
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}, loss: {epoch_loss_acc[0].item() / num_batches}, acc: {epoch_loss_acc[1].item()}")

        # validation loop
        with torch.inference_mode():
            epoch_loss_acc = torch.zeros((2,), device=device)
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                local_bs = target.shape[0]

                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                epoch_loss_acc[0] += loss * local_bs
                epoch_loss_acc[1] += (output.argmax(dim=1) == target).float().mean() * local_bs

            dist.all_reduce(epoch_loss_acc.data, op=dist.ReduceOp.SUM)
            epoch_loss_acc /= len(valid_dataset)
            if dist.get_rank() == 0:
                print(f"Rank {dist.get_rank()}, loss: {epoch_loss_acc[0].item() / num_batches}, acc: {epoch_loss_acc[1].item()}")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str)
    args = parser.parse_args()
    init_process(local_rank, fn=run_training, device_type=args.device)
