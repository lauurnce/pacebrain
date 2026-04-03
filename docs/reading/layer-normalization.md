# Layer Normalization — Ba et al., 2016

Normalises across features within each sample rather than across the batch.

Removes the batch dependence entirely, so batch size is irrelevant and training
and inference behave identically — which is why sequence models and transformers
adopted it over batch norm.

The comparison to batch norm makes the underlying question clear: what you
normalise over decides what your method can be used on.
