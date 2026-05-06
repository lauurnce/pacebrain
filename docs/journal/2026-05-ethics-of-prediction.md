# Study notes — where prediction stops being neutral

A model that predicts well can still be the wrong thing to deploy.

**Proxies.** The target is rarely the thing actually cared about. Predicting
who gets rearrested is not predicting who commits crime; it is predicting
policing. The model learns the proxy, and the gap between proxy and intent is
where harm lives.

**Feedback loops.** A model that influences the world it predicts changes the
distribution it was fitted to. Recommendations shape what gets watched, which
becomes the next training set.

**Aggregate accuracy hides subgroup failure.** A model can be accurate overall
and much worse for a minority group, and the aggregate is precisely the number
that conceals this. Per-group evaluation is the minimum.

**Confidence is a claim.** Reporting a single number implies a precision the
model does not have, and users act on that implied precision.

## The applicable version here

For a running predictor the stakes are low, but the habits transfer: say what
data it saw, report per-slice results, and give intervals rather than points
wherever someone might act on the number.
