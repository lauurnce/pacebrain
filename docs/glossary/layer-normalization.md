# Layer normalization

Normalising each sample across its own features rather than across the batch. Because no cross-sample statistics are involved, batch size is irrelevant and training and inference behave identically — which is why sequence models and transformers prefer it to [[batch-normalization]].
