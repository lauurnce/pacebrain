# Sigmoid

A function squashing any real number into (0, 1), used for binary probabilities and as a gate inside recurrent cells. Its derivative peaks at 0.25 and vanishes at both ends, which is why stacking many sigmoid layers kills the gradient and why ReLU displaced it in deep networks.

## See also

- [Vanishing gradient](vanishing-gradient.md)
- [Embedding](embedding.md)

The failure mode to watch with Sigmoid is silent optimism: results improve on the validation set for reasons that will not survive contact with next month's data.
