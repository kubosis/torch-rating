import math

import torch
from torch import nn
from torch.nn import functional as F


def inverse_softplus(value: float) -> float:
    """Return x such that softplus(x) equals value; require finite value > 0."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a finite positive value.")
    return value + math.log(-math.expm1(-value))


class PositiveScalar(nn.Module):
    """Scalar parameter exposed as softplus(raw) + eps, optionally trainable."""

    def __init__(
        self,
        value: float,
        learnable: bool,
        eps: float = 1e-6,
    ):
        """Initialize at a finite value > eps; learnable controls raw's gradients."""
        super().__init__()

        if value <= eps:
            raise ValueError(f"value must be greater than {eps}")

        self.eps = eps
        # Invert the transform so the exposed scalar starts at the requested value.
        self.raw = nn.Parameter(
            torch.tensor(
                inverse_softplus(value - eps),
                dtype=torch.float32,
            ),
            requires_grad=learnable,
        )

    def forward(self) -> torch.Tensor:
        """Return the constrained scalar tensor; call the module as scalar()."""
        return F.softplus(self.raw) + self.eps


class RatingAdapter(nn.Module):
    """Base for embedding-to-rating readouts; subclasses define rating_features."""

    def __init__(
        self,
        node_dim: int,
        rating_dim: int,
        raw_out_dim: int,
        out_dim: int,
    ):
        """Set input, projected rating, raw feature, and encoded output widths."""
        super().__init__()

        if min(node_dim, rating_dim, raw_out_dim, out_dim) <= 0:
            raise ValueError("All dimensions must be positive.")

        self.node_dim = node_dim
        self.rating_dim = rating_dim
        self.raw_out_dim = raw_out_dim
        self.out_dim = out_dim

        # Both teams use the same projection and rating coordinate system.
        self.project = nn.Linear(
            node_dim,
            rating_dim,
            bias=False,
        )

        self.encode = nn.Sequential(
            nn.Linear(raw_out_dim, out_dim),
            nn.SiLU(),
        )

    def rating_features(
        self,
        home_rating: torch.Tensor,
        away_rating: torch.Tensor,
    ) -> torch.Tensor:
        """Override: map two [..., rating_dim] tensors to [..., raw_out_dim]."""
        raise NotImplementedError

    def raw_features(
        self,
        home: torch.Tensor,
        away: torch.Tensor,
    ) -> torch.Tensor:
        """Validate [..., node_dim] inputs and return unencoded rating features."""
        if home.shape != away.shape:
            raise ValueError("Home and away shapes must match.")

        if home.ndim < 1 or home.shape[-1] != self.node_dim:
            raise ValueError(
                f"Expected inputs shaped [..., {self.node_dim}]."
            )

        return self.rating_features(
            self.project(home),
            self.project(away),
        )

    def forward(
        self,
        home: torch.Tensor,
        away: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw rating features with Linear + SiLU into [..., out_dim]."""
        return self.encode(self.raw_features(home, away))
