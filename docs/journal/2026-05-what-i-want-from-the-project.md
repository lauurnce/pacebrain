# Study notes — deciding what the project is for

Before writing code, worth being explicit about which goal is primary, because
they pull in different directions.

**Learning the tools.** Then the model should be simple and the pipeline
complete: data loading, training, evaluation, checkpointing, inference,
deployment. Breadth over depth.

**A real prediction.** Then almost all effort goes into data quality and honest
evaluation, and the model choice barely matters.

**A portfolio piece.** Then it has to be readable and the reasoning has to be
visible, which mostly means writing down why things were done, including the
things that failed.

## The choice

Primarily the first, with the third as a constraint — so: real pipeline,
modest models, and every decision written down as it happens rather than
reconstructed later.

## The commitment that follows

Synthetic data to start, so the ground truth is known and the evaluation can be
checked against it. That makes every early result a statement about the code
rather than about running, and that distinction has to stay stated or it
quietly stops being true.
