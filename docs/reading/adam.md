# Adam — Kingma & Ba, 2014

Combines momentum with per-parameter step sizes from a running estimate of
squared gradients, plus a bias correction for the early steps when both
estimates start at zero.

Its real contribution is robustness: it works acceptably across problems without
much tuning, which is why it became the default.

The bias correction is easy to skip when reimplementing and matters most in
exactly the first few hundred steps, where a wrong step size does lasting damage.
