# Batch normalization

Normalising each feature using statistics computed across the batch, then applying a learned scale and shift. It couples every sample to the others in its batch, which is why small batches destabilise it and why training and inference follow different code paths — inference uses running averages instead.
