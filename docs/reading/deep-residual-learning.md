# Deep Residual Learning — He et al., 2015

Adds skip connections so each block learns a residual correction rather than a
full transformation.

The observation that motivates it: deeper plain networks were performing *worse*
on training error, not just validation. That rules out overfitting and points at
an optimisation problem. If the extra layers could easily learn the identity,
depth should never hurt — and the skip connection makes the identity the default.

Enabled networks over a hundred layers deep where previous practice stalled far
earlier.
