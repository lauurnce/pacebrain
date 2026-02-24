# Activation checkpointing

Discarding intermediate activations during the forward pass and recomputing them during the backward pass. It trades compute for memory, which is what makes models larger than memory trainable at all.
