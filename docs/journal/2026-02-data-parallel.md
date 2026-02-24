# Data parallelism

Replicating the model across devices, splitting the batch, and averaging gradients. Simple and effective until the model itself exceeds one device, at which point the model has to be split instead.
