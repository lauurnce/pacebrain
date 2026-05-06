# Study notes — writing down what a model is

A checkpoint is not self-describing. A file of weights says nothing about what
it was trained on, what it is for, or where it fails.

## What a model card covers

Intended use, and explicitly the uses it is not suitable for. Training data and
its known biases. Evaluation results with the conditions they were measured
under. Limitations and failure modes. Ethical considerations where decisions
affect people.

## The section that gets skipped and matters most

Out-of-scope use. A model trained on one population and applied to another
fails quietly and confidently, and the only thing standing between that and a
bad decision is somebody having written down where the boundary is.

## Scaling it down

For a personal project this is a page. Even then it forces the useful question:
what would I have to say if someone relied on this? If the honest answer is
"do not", that belongs at the top in bold.
