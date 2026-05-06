# Vanishing gradient

When gradients shrink toward zero as they propagate backwards, so early layers stop learning while later ones train normally. It follows from the chain rule multiplying many small derivatives, which is why saturating activations and long recurrent sequences are the usual causes and why gates, residual connections and ReLU all exist as answers to it.
