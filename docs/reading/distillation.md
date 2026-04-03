# Distilling the Knowledge in a Neural Network — Hinton et al., 2015

Trains a small model on the soft output distribution of a large one.

The insight is that the soft targets carry information hard labels discard:
relative similarity between classes. A model told the answer is "dog" learns
less than one told it is mostly dog, somewhat cat, definitely not truck.

Explains why a student can outperform the same architecture trained on labels
alone, which otherwise looks impossible.
