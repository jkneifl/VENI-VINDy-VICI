import torch


def batch_jacobian(y, x, create_graph=False):
    """
    Compute batch Jacobian of y with respect to x.

    Parameters
    ----------
    y : torch.Tensor
        Output tensor of shape (batch, m).
    x : torch.Tensor
        Input tensor of shape (batch, n) that requires grad.
    create_graph : bool, optional
        If True, the Jacobian will be differentiable (needed for higher-order derivatives).

    Returns
    -------
    torch.Tensor
        Jacobian of shape (batch, m, n).
    """
    batch, m = y.shape
    jac = []
    for i in range(m):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, i] = 1.0
        g = torch.autograd.grad(
            y, x, grad_outputs=grad_outputs,
            create_graph=create_graph, retain_graph=True
        )[0]
        jac.append(g.unsqueeze(1))
    return torch.cat(jac, dim=1)


def batch_hessian(jac, x, create_graph=False):
    """
    Compute batch Hessian given a batch Jacobian.

    Given jac = dz/dx of shape (batch, m, n) and x of shape (batch, n),
    computes d^2z/dx^2 of shape (batch, m, n, n).

    Parameters
    ----------
    jac : torch.Tensor
        Jacobian tensor of shape (batch, m, n).
    x : torch.Tensor
        Input tensor of shape (batch, n) that requires grad.
    create_graph : bool, optional
        If True, the result will be differentiable.

    Returns
    -------
    torch.Tensor
        Hessian of shape (batch, m, n, n).
    """
    batch, m, n = jac.shape
    hess = []
    for i in range(m):
        # jac[:, i, :] has shape (batch, n) - compute its Jacobian wrt x
        jac_i = jac[:, i, :]  # (batch, n)
        hess_i = batch_jacobian(jac_i, x, create_graph=create_graph)  # (batch, n, n)
        hess.append(hess_i.unsqueeze(1))  # (batch, 1, n, n)
    return torch.cat(hess, dim=1)  # (batch, m, n, n)
