# Momentum

An optimiser term that accumulates a running average of past gradients so the step follows a smoothed direction rather than the latest noisy one. It damps oscillation across a narrow valley while accelerating along its floor, which is why it converges faster than plain SGD on badly conditioned surfaces.
