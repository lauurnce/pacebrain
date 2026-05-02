# Problem statement

Written from the PacePack side, before deciding whether a model is even the
right answer.

## The question users actually ask

"What should I target for this race?" Not "what is my predicted time" — the
prediction is only a means to picking a pace.

That distinction matters more than it looks. A prediction 5 minutes optimistic
on a marathon sends someone out about 7 sec/km too fast, and the back half
falls apart. The cost of error is **asymmetric**: predicting too fast is much
worse than predicting too slow, because one produces a blow-up and the other
produces a negative split and a good day.

Any loss function that treats those symmetrically is not modelling the actual
cost. Worth revisiting once there is something to tune.

## Why a formula alone is not enough

Riegel and its relatives need a recent race result. Most people asking the
question have not raced recently — that is usually *why* they are asking. So
the formula is unavailable exactly when it is most wanted.

What people do have is training data: months of it, logged automatically and
already sitting in the app. The opportunity is mapping training history to race
outcome, which is a different input space than any of the classic formulas use.

## What "good" would look like

- **Within 3-5 minutes on a marathon.** Roughly the width of a sensible pace
  band, so anything tighter is false precision given day-to-day variation in
  sleep, heat and fuelling.
- **Honest about uncertainty.** A confident single number for someone with two
  months of data is worse than a range.
- **Refuses to answer when it should.** No recent long runs plus a marathon
  target is not a prediction problem, it is a "please do not do this" problem.

## What would make this not worth building

- If a linear regression on the same features does as well, the network is
  decoration. This has to be checked, not assumed.
- If the honest error band is so wide it does not narrow anyone's pace choice,
  the output is not actionable regardless of how good the metric looks.

## Non-goals

Training plan generation, injury prediction, and anything needing real-time
data during a race. Each is a separate problem, and none is required to answer
the pacing question.
