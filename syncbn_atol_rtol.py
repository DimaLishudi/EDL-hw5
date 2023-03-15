import os

import torch
import torch.distributed as dist
import torch.nn as nn

from syncbn import SyncBatchNorm
import argparse
import pandas as pd
from collections import defaultdict

torch.set_num_threads(1)


def init_process(local_rank):
    """Initialize the distributed environment."""
    dist.init_process_group("gloo", rank=local_rank)
    run_experiments(local_rank)



def get_atol_rtol(local_rank, batch_size, n_features):
    torch.manual_seed(0)
    custom_bn = SyncBatchNorm(n_features)
    reference_bn = nn.BatchNorm1d(n_features, affine=False)
    torch.manual_seed(local_rank)
    data1 = torch.randn(batch_size//dist.get_world_size(), n_features)
    data2 = torch.clone(data1).detach()
    data1.requires_grad_()
    data2.requires_grad_()

    custom_forward = custom_bn(data1)
    reference_forward = reference_bn(data2)
    custom_forward.sum().backward()
    reference_forward.sum().backward()
    custom_backward = data1.grad.data
    reference_backward = data2.grad.data
    forward_atol = torch.max(torch.abs(custom_forward - reference_forward))
    forward_rtol = torch.max(torch.abs((custom_forward - reference_forward) / custom_forward))
    backward_atol = torch.max(torch.abs(custom_backward - reference_backward))
    backward_rtol = torch.max(torch.abs((custom_backward - reference_backward) / custom_backward))
    return forward_atol, forward_rtol, backward_atol, backward_rtol

def run_experiments(local_rank):
    batch_size_list = [32, 64]
    features_list = [128, 256, 512, 1024]

    for i, batch_size in enumerate(batch_size_list):
        results = defaultdict(dict)
        for n_features in features_list:
            forward_atol, forward_rtol, backward_atol, backward_rtol = get_atol_rtol(local_rank, batch_size, n_features)
            results[f"n_features={n_features}"]["forward_atol"] = forward_atol.detach().item()
            results[f"n_features={n_features}"]["forward_rtol"] = forward_rtol.detach().item()
            results[f"n_features={n_features}"]["backward_atol"] = backward_atol.detach().item()
            results[f"n_features={n_features}"]["backward_rtol"] = backward_rtol.detach().item()
        if local_rank == 0:
            print("Batch size:", batch_size)
            print(pd.DataFrame.from_dict(results))
            print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_process(local_rank)
