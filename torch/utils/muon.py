import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import List


# Grabed form the original paper: Noah Amsen et al. 2025
polar_express_coeffs = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375)
]

# For numerical stability
polar_express_coeffs = [
    (a / 1.01 , b / 1.01**3 , c / 1.01**5) for (a , b , c)
    # The last coefficient will repeat for rest of the steps
    in polar_express_coeffs[:-1]
] + [polar_express_coeffs[-1]]


@torch.compile(dynamic=False, fullgraph=True)
@torch.no_grad()
def _polar_express(G: torch.Tensor, steps: int = 5):
    '''
    Polar Express method for matrix orthogonalization.

    Source: https://arxiv.org/pdf/2505.16932 (Noah Amsen et al.)
    '''

    assert G.ndim >= 2
    X = G.bfloat16()

    # Keep larger dimension on the column side, to reduce symmetrical matrix
    # size, that is to reduce the size of X @ X.T. Reduces FLOPS required.
    if G.shape[-2] > G.shape[-1]:
        X = X.mT

    # Unit norm for stabilization
    X = X / (torch.linalg.norm(
        X,
        ord='fro', dim=(-2, -1), keepdims=True
    ) * 1.01 + 1e-9)

    # perform steps e.g. f ◌ f ◌ f ◌ ... ◌ f(X) for quintic polynomial
    for s in range(steps):
        s = 7 if s >= 7 else s  # Repeat final coefficient
        a, b, c = polar_express_coeffs[s]

        # X = aX + bX^3 + cX^5
        A = X @ X.mT  # `A` should be symmetrical
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.shape[-2] > G.shape[-1]:
        X = X.mT

    return X


class Muon(Optimizer):
    ''' The Muon Optimizer (single device). '''

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ortho_steps: int = 5
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ortho_steps=ortho_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group['params']

            # Reduce lookups overhead
            momentum = group['momentum']
            nesterov = group['nesterov']
            ortho_steps = group['ortho_steps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            for p in params:
                # parameter gradient
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if 'momentum_buf' not in state:
                    state['momentum_buf'] = torch.zeros_like(g)
                buf = state['momentum_buf']

                # buf = momentum * buf + (1 - momentum) * g
                buf.lerp_(g, 1 - momentum)

                # Nesterov momentum
                g = g.lerp_(buf, momentum) if nesterov else buf

                # Orthogonalize
                g = _polar_express(g, steps=ortho_steps)

                # Apply weight decay
                # p = p - lr * weight_decay * p
                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                # Update
                p.sub_(g.view_as(p), alpha=lr)

        return loss


class DistributedMuon(Optimizer):
    ''' The Distributed Muon Optimizer. '''

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ortho_steps: int = 5
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ortho_steps=ortho_steps
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        for group in self.param_groups:
            params = group['params']
            nparams = len(params)  # non padded length
            # Make parameter count divisible by world size.
            params = params + [ torch.zeros_like(params[-1]) ] * (world_size -
                nparams % world_size)

            # Reduce lookups overhead
            momentum = group['momentum']
            nesterov = group['nesterov']
            ortho_steps = group['ortho_steps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            for i in range(0, nparams, world_size):
                # No need to update for padded parameters
                if i + rank < nparams:
                    # parameter and gradient
                    p = params[i + rank]
                    g = p.grad
                    if g is None:
                        # Force update for syncrhonization (unlikely)
                        g = torch.zeros_like(p)

                    state = self.state[p]
                    if 'momentum_buf' not in state:
                        state['momentum_buf'] = torch.zeros_like(g)
                    buf = state['momentum_buf']

                    # buf = momentum * buf + (1 - momentum) * g
                    buf.lerp_(g, 1 - momentum)

                    # Nesterov momentum
                    g = g.lerp_(buf, momentum) if nesterov else buf

                    # Orthogonalize
                    g = _polar_express(g, steps=ortho_steps)

                    # Apply weight decay
                    # p = p - lr * weight_decay * p
                    if weight_decay != 0.0:
                        p.mul_(1 - lr * weight_decay)

                    # Update
                    p.sub_(g.view_as(p), alpha=lr)

                # Broadcast the updated parameter and gather other updated
                # parameters from other processes.
                dist.all_gather(params[i:i+world_size], params[i + rank])

        return loss
