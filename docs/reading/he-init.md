# Delving Deep into Rectifiers — He et al., 2015

Adapts variance-preserving initialisation for ReLU, where zeroing negative
inputs halves the variance passed forward, and introduces PReLU.

The factor of two in the scale is the whole correction, and it follows directly
from the previous paper's derivation once the activation changes.

This is the right default for ReLU networks and the reason frameworks ship it.
