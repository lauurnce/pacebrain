# Batch Normalization — Ioffe & Szegedy, 2015

Normalises layer inputs using batch statistics, with a learned scale and shift.

Allowed much higher learning rates and reduced sensitivity to initialisation.
The stated justification was reducing "internal covariate shift".

Worth reading alongside the later work disputing that explanation — the
technique works, the mechanism given for it does not appear to be the real one.
A clean example of an empirical result and its story being separable claims.
