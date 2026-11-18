import torch.nn as nn
import torch

from torch_rating._rating_layers._adapter import PositiveScalar, RatingAdapter


class EloLayer(nn.Module):
    """
    Elo with autograd numerical backward pass

    Loss function should be weighted MSE, where Weight is the goal difference in the match raised to the power of
    EloAutoGrad.gamma parameter or CrossEntropyLoss. In case of CrossEntropyLoss the goal difference is
    not accounted for.

    Outputs tensor of same shape as Input

    Forward Computation:
    $E_H = \frac{1}{1 + c^{\frac{a_i - h_i}{d}}}$

    $E_A = 1 - E_H$

    Where:
    - $h_i$: home team rating
    - $a_i$: away team rating
    """

    def __init__(
            self,
            in_channels: int,
            gamma: float = 2.0,
            c: float = 10.0,
            d: float = 400.,
            hp_grad: bool = False,
    ):
        """
        :param gamma: (float) impact scale of goal difference, default value = 2.
        :param c: (float) rating meta parameter, default value = 3.
        :param d: (float) rating meta parameter, default value = 0.8.
        :param hp_grad: (bool) whether to use gradient descend for ratings hyperparams
        """
        assert in_channels > 0

        super(EloLayer, self).__init__()
        self.gamma = gamma
        self.in_channels = in_channels
        self.c = nn.Parameter(torch.tensor(c, dtype=torch.float), requires_grad=hp_grad)
        self.d = nn.Parameter(torch.tensor(d, dtype=torch.float), requires_grad=hp_grad)

    def forward(self, home, away):
        assert home.shape == away.shape
        assert len(home) == self.in_channels

        E_H = 1 / (1 + torch.pow(self.c, ((away - home) / self.d)))

        E_A = 1 - E_H
        return torch.cat([E_A, E_H], dim=0)


class EloAdapter(RatingAdapter):
    """Elo-inspired probability features from latent rating differences."""

    def __init__(
        self,
        node_dim: int,
        rating_dim: int = 8,
        out_dim: int = 16,
        learnable_parameters: bool = True,
        initial_scale: float = 1.0,
    ):
        """Set embedding widths and positive sigmoid scale; optionally train scale."""
        super().__init__(
            node_dim=node_dim,
            rating_dim=rating_dim,
            raw_out_dim=2 * rating_dim,
            out_dim=out_dim,
        )

        self.scale = PositiveScalar(
            initial_scale,
            learnable=learnable_parameters,
        )

    def rating_features(self, home_rating, away_rating):
        """Return away/home probabilities shaped [..., 2 * rating_dim]."""
        home_probability = torch.sigmoid(
            self.scale() * (home_rating - away_rating)
        )

        # Preserve the original package's away-first ordering.
        return torch.cat(
            [1.0 - home_probability, home_probability],
            dim=-1,
        )
