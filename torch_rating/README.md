# Rating adapters

Run the CPU regression checks from the repository root with
`python -m unittest discover -s tests -v`.

`_rating_layers/_adapter.py` provides shared building blocks for `EloAdapter`,
`BerrarAdapter`, and `PiAdapter`. These modules turn home/away embeddings into
features for a downstream prediction head. They do not store team ratings or
update match history.

## Using an adapter

```python
import torch
from torch_rating import EloAdapter

adapter = EloAdapter(node_dim=32, rating_dim=8, out_dim=16)
home, away = torch.randn(4, 32), torch.randn(4, 32)
raw = adapter.raw_features(home, away)  # (4, 16): away/home probabilities
features = adapter(home, away)         # (4, 16): learned encoded features
```

Inputs must have identical shapes `[..., node_dim]`, with floating-point dtype
and device matching the module. Leading dimensions are preserved. Encoded
features are not probabilities or scores; use `raw_features()` for those.

## `RatingAdapter`

Subclass this base and implement `rating_features()`. All constructor dimensions
must be positive:

| Argument | Meaning |
| --- | --- |
| `node_dim` | Input embedding width |
| `rating_dim` | Projected rating width per team |
| `raw_out_dim` | Width returned by the subclass's rating formula |
| `out_dim` | Encoded feature width |

- `project`: shared, bias-free `Linear(node_dim, rating_dim)` for both teams.
- `rating_features(home_rating, away_rating)`: subclass hook taking two
  `[..., rating_dim]` tensors and returning `[..., raw_out_dim]`. The base raises
  `NotImplementedError`; direct calls bypass input validation and projection.
- `raw_features(home, away)`: validates matching input shapes, projects both
  embeddings, and calls `rating_features()` without encoding.
- `forward(home, away)`: called through `adapter(home, away)`; applies `encode`
  (`Linear(raw_out_dim, out_dim)` followed by `SiLU`) to the raw features.
  `BerrarAdapter` overrides this to divide raw scores by `feature_scale` first.

Existing adapters concatenate away features before home features on the last
axis. Elo produces `2 * rating_dim` raw probabilities; Berrar and Pi produce
`rating_dim` raw scores and margins respectively and require even `rating_dim`.
Projection and encoding remain trainable when `learnable_parameters=False`;
that flag freezes only the rating-formula parameters.

## Individual adapters

Import `EloAdapter`, `BerrarAdapter`, or `PiAdapter` from `torch_rating` and use
the same calling pattern as above. Each requires `node_dim` and defaults to
`rating_dim=8`, `out_dim=16`, and `learnable_parameters=True`. All use ordinary
autograd. Below, `h` and `a` are the projected ratings, and `sigmoid` acts
elementwise. Positive scalar initializers must be finite and greater than `1e-6`.

### `EloAdapter`

Use for pairwise win-probability features. `rating_features(h, a)` computes
`p_home = sigmoid(scale * (h - a))` and returns `[1 - p_home, p_home]`, shaped
`[..., 2 * rating_dim]`. Each channel pair sums to one; there is no draw output.

`initial_scale=1.0` initializes the positive `scale`; larger values make
probabilities more sensitive to rating differences. This plays the role of
`log(c) / d` in the classical Elo formula. `forward()` encodes these probabilities
into `[..., out_dim]` features.

### `BerrarAdapter`

Use for score features based on attack and opponent defensive weakness.
`rating_dim` must be even: each projected vector splits into `[attack, weakness]`.
`rating_features(h, a)` computes:

```text
home_score = alpha_h * sigmoid(beta_h * (h_attack + a_weakness) + bias_h)
away_score = alpha_a * sigmoid(beta_a * (a_attack + h_weakness) + bias_a)
```

It returns `[away_score, home_score]`, shaped `[..., rating_dim]`. Greater
weakness increases the opponent's score. Positive `alpha_h/alpha_a=80.0` set
score ceilings, positive `beta_h/beta_a=1.0` control steepness, and
`bias_h/bias_a=0.0` shift the sigmoid inputs. Adjust these defaults to the sport.

`forward()` divides raw scores by the fixed `feature_scale=80.0` before encoding
to `[..., out_dim]`. This scale must be finite and positive; `raw_features()`
always returns unscaled scores.

### `PiAdapter`

Use for signed score-margin features. `rating_dim` must be even: each projected
vector splits into `[home_rating, away_rating]`. Only the home team's first half
and the away team's second half contribute to a match.

`signed_score(rating)` preserves shape and converts ratings to signed scores:
`z = M * tanh(log(10) * rating / (c * M))`, then `sign(z) * expm1(abs(z))`.
The implementation preserves a nonzero derivative at zero. Positive `c=3.0`
controls rating sensitivity; fixed `M = max_exponent=3.0` must lie in `(0, 10]`
and smoothly limits exponential growth, modifying the classical Pi formula.

`rating_features(h, a)` computes
`margin = signed_score(h_home) - signed_score(a_away)` and returns
`[-margin, margin]`, shaped `[..., rating_dim]`. Positive home margins favor the
home team. `forward()` encodes these into `[..., out_dim]` features. This adapter
does not apply classical Pi rating updates or use `lambda_`/`gamma`.

## `PositiveScalar` and `inverse_softplus`

`PositiveScalar(value, learnable, eps=1e-6)` stores an unconstrained scalar
`nn.Parameter` named `raw`. Calling it, e.g. `scale = PositiveScalar(1.0, True)`
then `scale()`, returns `softplus(raw) + eps`. This keeps scales positive during
optimization. The initial `value` must be finite and greater than `eps`; use a
positive `eps` for a positive floor. `learnable=False` disables gradients for
`raw`, which remains registered for checkpoints and module device/dtype moves.

`inverse_softplus(value)` returns a Python float whose softplus equals `value`
and raises `ValueError` for nonfinite or nonpositive inputs. The constructor uses
`inverse_softplus(value - eps)` to initialize the exposed scalar at `value`.
