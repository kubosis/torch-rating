import copy
import math

import torch.nn as nn
import torch

from torch_rating._rating_layers._adapter import PositiveScalar, RatingAdapter


class PiLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 lambda_: float = 0.1,
                 gamma: float = 1.,
                 c: float = 3.,
                 ):
        assert in_channels % 2 == 0 and in_channels > 0
        super(PiLayer, self).__init__()

        self.in_channels = in_channels

        self.c = torch.tensor(c, dtype=torch.float)
        self.lambda_ = torch.tensor(lambda_, dtype=torch.float)
        self.gamma = torch.tensor(gamma, dtype=torch.float)

    def forward(self, home, away):
        assert len(home) == len(away) == self.in_channels
        return _pi_fwd(home, away, self.c, self.lambda_, self.gamma)


class _PiFunction(torch.autograd.Function):
    @staticmethod
    def forward(home, away, c, lambda_, gamma):
        h_half, a_half = len(home) // 2, len(away) // 2

        R_alpha_h = home[:h_half]
        R_beta_a = away[a_half:]

        g_hat_da = torch.pow(10, torch.abs(R_beta_a) / c) - 1
        g_hat_dh = torch.pow(10, torch.abs(R_alpha_h) / c) - 1

        g_hat_da *= torch.where(R_beta_a < 0, torch.tensor([-1], device=g_hat_da.device), torch.tensor([1], device=g_hat_da.device))
        g_hat_dh *= torch.where(R_alpha_h < 0, torch.tensor([-1], device=g_hat_da.device), torch.tensor([1], device=g_hat_da.device))

        g_hat_away = g_hat_da - g_hat_dh
        g_hat_home = g_hat_dh - g_hat_da

        return torch.cat([g_hat_away, g_hat_home], dim=0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        home, away, c, lambda_, gamma = inputs
        ctx.save_for_backward(home, away, c, lambda_, gamma, output)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        home, away, c, lambda_, gamma, prediction = ctx.saved_tensors

        h_half, a_half = len(home) // 2, len(away) // 2

        R_alpha_h, R_alpha_a = home[:h_half], home[h_half:]
        R_beta_h, R_beta_a = away[:a_half], away[a_half:]

        g_out_half = len(prediction) // 2

        error_a, error_h = prediction[:g_out_half], prediction[g_out_half:]

        R_alpha_new_h = R_alpha_h + lambda_ * c * torch.log10(1 + torch.abs(error_h))
        R_alpha_new_a = R_alpha_a + gamma * (R_alpha_new_h - R_alpha_h)

        R_beta_new_a = R_beta_a + lambda_ * c * torch.log10(1 + torch.abs(error_a))
        R_beta_new_h = R_beta_h + gamma * (R_beta_new_a - R_alpha_a)

        grad_home = torch.cat([R_alpha_new_h, R_alpha_new_a], dim=0)
        grad_away = torch.cat([R_beta_new_h, R_beta_new_a], dim=0)

        return grad_home, grad_away, None, None, None


# create alias
_pi_fwd = _PiFunction.apply


class PiAdapter(RatingAdapter):
    """Pi-inspired margin readout using ordinary autograd.

    This is a readout, not the classical Pi state-update rule.
    """

    def __init__(
        self,
        node_dim: int,
        rating_dim: int = 8,
        out_dim: int = 16,
        learnable_parameters: bool = True,
        c: float = 3.0,
        max_exponent: float = 3.0,
    ):
        """Set even rating width, positive divisor c, and fixed exponent bound."""
        if rating_dim <= 0 or rating_dim % 2:
            raise ValueError("Pi rating_dim must be positive and even.")
        if not 0 < max_exponent <= 10:
            raise ValueError("max_exponent must be in (0, 10].")

        super().__init__(
            node_dim=node_dim,
            rating_dim=rating_dim,
            raw_out_dim=rating_dim,
            out_dim=out_dim,
        )

        self.c = PositiveScalar(c, learnable_parameters)

        self.register_buffer(
            "max_exponent",
            torch.tensor(float(max_exponent)),
        )

    def signed_score(self, rating):
        """Map ratings to signed scores with bounded exponents; preserve shape."""
        exponent = math.log(10.0) * rating / self.c()

        # Smoothly bound the exponential argument.
        # This deliberately modifies the original formula at large magnitudes.
        exponent = self.max_exponent * torch.tanh(
            exponent / self.max_exponent
        )

        # Both branches have derivative 1 at zero.
        # Unlike sign(x) * expm1(abs(x)), this preserves that gradient.
        return torch.where(
            exponent >= 0,
            torch.expm1(exponent),
            -torch.expm1(-exponent),
        )

    def rating_features(self, home_rating, away_rating):
        """Return opposite away/home margins shaped [..., rating_dim]."""
        # Select each team's rating for its role in this match.
        home_at_home, _ = home_rating.chunk(2, dim=-1)
        _, away_away = away_rating.chunk(2, dim=-1)

        home_margin = (
            self.signed_score(home_at_home) - self.signed_score(away_away)
        )

        return torch.cat(
            [-home_margin, home_margin],
            dim=-1,
        )
