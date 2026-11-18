import math

import torch.nn as nn
import torch

from torch_rating._rating_layers._adapter import PositiveScalar, RatingAdapter


class BerrarLayer(nn.Module):
    """
    Berra with gradient with numerical backward pass via torch autograd system

    Loss function should be MSE, forward pass outputs predicted
    number of goals scored by [home, away] team in Tensor
    """

    def __init__(
        self,
        in_channels: int = 2,
        alpha_h: float = 80.0,  # basketball 80, football 3.6, am. football ?
        beta_h: float = 2.0,
        bias_h: float = 0.0,
        alpha_a: float = 80.0,
        beta_a: float = 2.0,
        bias_a: float = 0.0,
        hp_grad: bool = False,
    ):
        """
        note that default values are meant specifically for basketball

        :param alpha_h: expected number of goals by home team, default = 80
        :param beta_h: steepness of exponential for home team, default = 2
        :param bias_h:  bias of home team, default = 0
        :param alpha_a: expected number of goals by away team, default = 80
        :param beta_a: steepness of exponential for away team, default = 2
        :param bias_a: bias of away team, default = 0
        """
        assert in_channels % 2 == 0 and in_channels > 0

        super(BerrarLayer, self).__init__()

        self.in_channels = in_channels

        self.alpha_h = nn.Parameter(torch.tensor(alpha_h, dtype=torch.float, requires_grad=hp_grad))
        self.beta_h = nn.Parameter(torch.tensor(beta_h, dtype=torch.float, requires_grad=hp_grad))
        self.alpha_a = nn.Parameter(torch.tensor(alpha_a, dtype=torch.float, requires_grad=hp_grad))
        self.beta_a = nn.Parameter(torch.tensor(beta_a, dtype=torch.float, requires_grad=hp_grad))
        self.bias_h = nn.Parameter(torch.tensor(bias_h, dtype=torch.float, requires_grad=hp_grad))
        self.bias_a = nn.Parameter(torch.tensor(bias_a, dtype=torch.float, requires_grad=hp_grad))

    def forward(self, home, away):
        assert len(home) == len(away) == self.in_channels

        h_half, a_half = len(home) // 2, len(away) // 2
        hatt, hdef = home[:h_half], home[h_half:]
        aatt, adef = away[:a_half], away[a_half:]

        ah, bh, yh = self.alpha_h, self.beta_h, self.bias_h
        aa, ba, ya = self.alpha_a, self.beta_a, self.bias_a

        ghat_h = ah / (1 + torch.exp(-bh * (hatt + adef) - yh))
        ghat_a = aa / (1 + torch.exp(-ba * (aatt + hdef) - ya))

        return torch.cat([ghat_a, ghat_h], dim=0)


class BerrarAdapter(RatingAdapter):
    """Attack / defensive-weakness features with a Berrar readout."""

    def __init__(
        self,
        node_dim: int,
        rating_dim: int = 8,
        out_dim: int = 16,
        learnable_parameters: bool = True,
        alpha_h: float = 80.0,
        alpha_a: float = 80.0,
        beta_h: float = 1.0,
        beta_a: float = 1.0,
        bias_h: float = 0.0,
        bias_a: float = 0.0,
        feature_scale: float = 80.0,
    ):
        """Set even rating width, score ceilings/slopes/biases, and encoder scale."""
        if rating_dim <= 0 or rating_dim % 2:
            raise ValueError("Berrar rating_dim must be positive and even.")
        if not math.isfinite(feature_scale) or feature_scale <= 0:
            raise ValueError("feature_scale must be finite and positive.")

        super().__init__(
            node_dim=node_dim,
            rating_dim=rating_dim,
            raw_out_dim=rating_dim,
            out_dim=out_dim,
        )

        self.alpha_h = PositiveScalar(alpha_h, learnable_parameters)
        self.alpha_a = PositiveScalar(alpha_a, learnable_parameters)
        self.beta_h = PositiveScalar(beta_h, learnable_parameters)
        self.beta_a = PositiveScalar(beta_a, learnable_parameters)

        self.bias_h = nn.Parameter(
            torch.tensor(float(bias_h)),
            requires_grad=learnable_parameters,
        )
        self.bias_a = nn.Parameter(
            torch.tensor(float(bias_a)),
            requires_grad=learnable_parameters,
        )

        # Fixed numerical scaling before the learned feature encoder.
        self.register_buffer(
            "feature_scale",
            torch.tensor(float(feature_scale)),
        )

    def rating_features(self, home_rating, away_rating):
        """Return unscaled away/home scores shaped [..., rating_dim]."""
        # Higher defensive weakness raises the opponent's predicted score.
        home_attack, home_weakness = home_rating.chunk(2, dim=-1)
        away_attack, away_weakness = away_rating.chunk(2, dim=-1)

        home_score = self.alpha_h() * torch.sigmoid(
            self.beta_h() * (home_attack + away_weakness) + self.bias_h
        )
        away_score = self.alpha_a() * torch.sigmoid(
            self.beta_a() * (away_attack + home_weakness) + self.bias_a
        )

        return torch.cat([away_score, home_score], dim=-1)

    def forward(self, home, away):
        """Scale raw scores before encoding to [..., out_dim] features."""
        raw = self.raw_features(home, away)
        return self.encode(raw / self.feature_scale)
