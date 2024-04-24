import torch.nn as nn
import torch


class Berrar(nn.Module):
    """
    Berra with gradient with numerical backward pass via torch autograd system

    Loss function should be MSE, forward pass outputs predicted
    number of goals scored by [home, away] team in Tensor
    """
    def __init__(self, alpha_h: float = 80., beta_h: float = 2., bias_h: float = 0.,
                 alpha_a: float = 80., beta_a: float = 2., bias_a: float = 0., hp_grad: bool = False):
        """
        note that default values are meant specifically for basketball
        
        :param alpha_h: expected number of goals by home team, default = 80
        :param beta_h: steepness of exponential for home team, default = 2
        :param bias_h:  bias of home team, default = 0
        :param alpha_a: expected number of goals by away team, default = 80
        :param beta_a: steepness of exponential for away team, default = 2
        :param bias_a: bias of away team, default = 0
        """
        super(Berrar, self).__init__()
        self.softmax = nn.Softmax(dim=0)
        if not hp_grad:
            self.alpha_h = alpha_h
            self.beta_h = beta_h
            self.alpha_a = alpha_a
            self.beta_a = beta_a
            self.bias_h = bias_h
            self.bias_a = bias_a
        else:
            self.alpha_h = nn.Parameter(torch.tensor(alpha_h, dtype=torch.float))
            self.beta_h = nn.Parameter(torch.tensor(beta_h, dtype=torch.float))
            self.alpha_a = nn.Parameter(torch.tensor(alpha_a, dtype=torch.float))
            self.beta_a = nn.Parameter(torch.tensor(beta_a, dtype=torch.float))
            self.bias_h = nn.Parameter(torch.tensor(bias_h, dtype=torch.float))
            self.bias_a = nn.Parameter(torch.tensor(bias_a, dtype=torch.float))

    def forward(self, home, away):
        assert len(home) == len(away)
        assert len(home) % 2 == 0

        h_half, a_half = len(home) // 2, len(away) // 2
        hatt, hdef = home[:h_half], home[h_half:]
        aatt, adef = away[:a_half], away[:a_half]

        ah, bh, yh = self.alpha_h, self.beta_h, self.bias_h
        aa, ba, ya = self.alpha_a, self.beta_a, self.bias_a

        ghat_h = ah / (1 + torch.exp(-bh * (hatt + adef) - yh))
        ghat_a = aa / (1 + torch.exp(-ba * (aatt + hdef) - ya))

        y_hat = self.softmax(torch.cat([ghat_a, ghat_h], dim=0))

        return y_hat
