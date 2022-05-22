import torch
import torch.nn as nn


class ConjunctiveYes(nn.Module):
    """
    alpha ^ beta -> gamma, where gamma is not conflict
    """

    def __init__(self, device):
        super(ConjunctiveYes, self).__init__()
        self.zero = torch.tensor(0, dtype=torch.float, requires_grad=False).to(device)

    def forward(self, alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx):
        return torch.max(self.zero, alpha[:, alpha_idx] + beta[:, beta_idx] - gamma[:, gamma_idx])


class ConjunctiveNot(nn.Module):
    """
    alpha ^ beta -> not delta, where delta is conflict
    """

    def __init__(self, device):
        super(ConjunctiveNot, self).__init__()
        self.zero = torch.tensor(0, dtype=torch.float, requires_grad=False).to(device)
        self.one = torch.tensor(1, dtype=torch.float, requires_grad=False).to(device)

    def forward(self, alpha, beta, gamma, alpha_idx, beta_idx, gamma_idx):
        very_small = 1e-8
        not_gamma = (self.one - gamma.exp()).clamp(very_small).log()
        return torch.max(self.zero, alpha[:, alpha_idx] + beta[:, beta_idx] - not_gamma[:, gamma_idx])


class SymmetryLoss(nn.Module):
    """
    alpha <-> beta, where beta is the opposite to alpha
    """

    def __init__(self):
        super(SymmetryLoss, self).__init__()

    def forward(self, alpha, beta, alpha_idx, beta_idx):
        return torch.abs(alpha[:, alpha_idx] - beta[:, beta_idx])




