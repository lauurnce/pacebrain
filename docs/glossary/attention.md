# Attention

A mechanism computing a weighted sum of values, where the weights come from the similarity between a query and a set of keys. It lets any position in a sequence draw directly on any other in one step, which removes the fixed-width bottleneck of a recurrent hidden state at the cost of being quadratic in sequence length.

## See also

- [Softmax](softmax.md)
- [Transformer](transformer.md)

The failure mode to watch with Attention is silent optimism: results improve on the validation set for reasons that will not survive contact with next month's data.
