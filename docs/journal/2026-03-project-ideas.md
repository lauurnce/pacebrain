# Candidate projects

Notes are only worth so much. Time to build something. Criteria, then options.

## Criteria

1. **A domain I already understand.** Debugging a model in an unfamiliar domain
   is guesswork — you cannot tell a wrong answer from a surprising one.
2. **Data I can actually obtain**, without an approval process or a scraping
   project.
3. **Small enough to train on a laptop CPU.** Waiting on a GPU budget is a way
   to not start.
4. **A real consumer for the output.** A model nobody uses never gets the
   feedback that shows it is wrong.
5. **Room for both a tabular and a sequence model**, so the notes on RNNs get
   exercised rather than staying theoretical.

## Options considered

**Expense categorisation.** Data is available and the problem is real, but it
is a text classification task and I would learn NLP tooling more than
fundamentals. Fails criterion 5.

**Course grade prediction.** Data is obtainable and the domain is familiar.
Small sample, few features, and a linear model would almost certainly match
anything more complex — which makes it a poor showcase and an ethically
awkward thing to deploy. Fails 5.

**Sleep quality from wearable data.** Genuinely sequential and the data exists.
But I do not understand sleep physiology well enough to know when a prediction
is absurd. Fails 1.

**Race time and pacing prediction.** Running is the domain I know best. The
data is already logged automatically. Both models fall out naturally — a
tabular predictor for finish time, a sequence model for per-segment pace. And
it plugs directly into PacePack, so there is a real consumer.

Meets all five.

## Why it is not trivial

The obvious objection is that formulas for this already exist. But they all
require a recent race result as input, and the interesting case is precisely
the person who has not raced recently. Mapping *training history* to race
outcome is a different input space, and I have not found anyone doing it at
consumer scale.

## Known risk

The labels are expensive. You only learn the true finish time when someone
actually races, so a single runner contributes very few labelled examples
against months of training input. Worth thinking about properly before
committing — it may force starting with synthetic data to prove the pipeline
before any real dataset exists.

## Decision

Going with the running model. Next step is writing down the problem properly
from the product side, rather than starting from the model.
