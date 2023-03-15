import os

import torch
import torch.distributed as dist
import torch.nn as nn

from syncbn import SyncBatchNorm
import argparse
import matplotlib.pyplot as plt
torch.set_num_threads(1)


def init_process(local_rank):
    """Initialize the distributed environment."""
    dist.init_process_group("gloo", rank=local_rank)
    size = dist.get_world_size()
    run_experiments(local_rank)



def get_atol_rtol(local_rank, batch_size, n_features):
    torch.manual_seed(0)
    custom_bn = SyncBatchNorm(n_features)
    reference_bn = nn.BatchNorm1d(n_features, affine=False)
    torch.manual_seed(local_rank)
    data1 = torch.randn(batch_size//2, n_features)
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
    fig, axs = plt.subplots(figsize=(10, 12), nrows=2, ncols=2)
    batch_size_list = [32, 64]
    features_list = [128, 256, 512, 1024]
    styles = ["--", ":"]
    handles = []
    for i, batch_size in enumerate(batch_size_list):
        forward_atols, forward_rtols = [], []
        backward_atols, backward_rtols = [], []
        for n_features in features_list:
            forward_atol, forward_rtol, backward_atol, backward_rtol = get_atol_rtol(local_rank, batch_size, n_features)
            forward_atols.append(forward_atol.detach().numpy())
            forward_rtols.append(forward_rtol.detach().numpy())
            backward_atols.append(backward_atol.detach().numpy())
            backward_rtols.append(backward_rtol.detach().numpy())
        if local_rank == 0:
            print("START")
            axs[0][0].scatter(features_list, forward_atols, linestyle=styles[i])
            axs[0][0].set(xlabel="n_features", ylabel="forward atol")
            axs[0][1].scatter(features_list, forward_rtols, linestyle=styles[i])
            axs[0][1].set(xlabel="n_features", ylabel="forward rtol")
            axs[1][0].scatter(features_list, backward_atols, linestyle=styles[i])
            axs[1][0].set(xlabel="n_features", ylabel="backward atol")
            axs[1][1].scatter(features_list, backward_rtols, linestyle=styles[i])
            axs[1][1].set(xlabel="n_features", ylabel="backward rtol")
    
        line = plt.Line2D([0], [0], label=f"batch_size = {batch_size}", color='k', linestyle=styles[i])
        handles.append(line)
    axs[0][0].legend(handles=handles)
    axs[0][1].legend(handles=handles)
    axs[1][0].legend(handles=handles)
    axs[1][1].legend(handles=handles)


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    init_process(local_rank)
