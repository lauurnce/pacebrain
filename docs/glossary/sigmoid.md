# Sigmoid

A function squashing any real number into (0, 1), used for binary probabilities and as a gate inside recurrent cells. Its derivative peaks at 0.25 and vanishes at both ends, which is why stacking many sigmoid layers kills the gradient and why ReLU displaced it in deep networks.
