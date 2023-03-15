import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_var, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        
        device = input.device
        batch_size, num_features = input.shape
        msg = torch.empty(2*num_features + 1, device=device)
        msg[0:num_features] = input.sum(dim=0)                   # first moment
        msg[num_features:2*num_features] = (input**2).sum(dim=0) # second moment
        msg[-1] = batch_size # local batch size

        dist.all_reduce(msg, dist.ReduceOp.SUM)
        
        mean = msg[0:num_features] / msg[-1]
        var = msg[num_features:2*num_features] / msg[-1] - mean**2
        sqrt_var = torch.sqrt(var + eps)

        t = input - mean
        res = t / sqrt_var

        running_mean = running_mean * (1 - momentum) + mean * momentum
        running_var = running_var * (1 - momentum) + var * momentum
        
        ctx.save_for_backward(t, sqrt_var, msg[-1])
        ctx.mark_non_differentiable(running_mean, running_var)
        return res, running_mean, running_var

    @staticmethod
    def backward(ctx, grad_output, grad_mean, grad_var):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        # Derivation of the following formulas are presented in report.ipynb
        # there g -- grad_output
        t, s, B = ctx.saved_tensors
        num_features = t.shape[1]

        device = grad_output.device
        msg = torch.empty(3*num_features, device=device)
        msg[:num_features] = grad_output.sum(dim=0)       # 1^T g
        msg[num_features:2*num_features] = t.sum(dim=0)   # 1^T t
        msg[2*num_features:] = (grad_output*t).sum(dim=0) # t^T g
        dist.all_reduce(msg, dist.ReduceOp.SUM)

        res = grad_output / s \
            - t * msg[2*num_features:] / (B * s**3) \
            - msg[:num_features] / (B * s) \
            + msg[num_features:2*num_features] * msg[2*num_features:] / (B**2 * s**3)
            
        return res, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        self.running_mean = torch.zeros((num_features,))
        self.running_var = torch.ones((num_features,)) # I use var instead of std for convenience
        self.bn_func = sync_batch_norm.apply
        # num_features, eps, momentum are saved in parent class

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training and self.track_running_stats:
            sqrt_var = torch.sqrt(self.running_var + self.eps)
            return (input - self.running_mean) / sqrt_var
        else:
            res, self.running_mean, self.running_var = self.bn_func(input, self.running_mean, self.running_var, self.eps, self.momentum)
            return res

